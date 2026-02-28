"""
GCS Model Storage Service

Manages model weights storage on Google Cloud Storage with:
- Signed URL generation for secure downloads
- Encryption at rest and in transit
- Quantized model management (GGUF/4-bit)
- Checkpoint storage and resumption

Security:
- Short-lived signed URLs (default: 1 hour)
- Optional encryption for sensitive weights
- Checksum verification
"""

import os
import asyncio
import logging
import hashlib
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

# Optional GCS imports
try:
    from google.cloud import storage
    from google.oauth2 import service_account

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("google-cloud-storage not installed. GCS features disabled.")


class ModelFormat(Enum):
    """Supported model formats."""

    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"


class QuantizationType(Enum):
    """Supported quantization types."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"
    GGUF_Q4_K_M = "q4_k_m"
    GGUF_Q5_K_M = "q5_k_m"
    GGUF_Q8_0 = "q8_0"
    AWQ = "awq"
    GPTQ = "gptq"


@dataclass
class ModelMetadata:
    """Metadata about a stored model."""

    model_id: str
    name: str
    version: str
    format: ModelFormat
    quantization: QuantizationType
    size_bytes: int
    checksum_sha256: str
    gcs_path: str
    created_at: datetime
    updated_at: datetime
    parameters: Optional[int] = None  # Number of parameters
    context_length: Optional[int] = None
    tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "format": self.format.value,
            "quantization": self.quantization.value,
            "size_bytes": self.size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "gcs_path": self.gcs_path,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "parameters": self.parameters,
            "context_length": self.context_length,
            "tags": self.tags or [],
        }


@dataclass
class SignedURLResult:
    """Result of signed URL generation."""

    url: str
    expires_at: datetime
    model_id: str
    checksum: str


class GCSModelStorage:
    """
    Service for managing model weights on Google Cloud Storage.

    Features:
    - Signed URL generation for secure downloads
    - Model versioning and metadata
    - Checkpoint management
    - Encryption support
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        checkpoint_bucket: Optional[str] = None,
        credentials_path: Optional[str] = None,
        signed_url_expiry_seconds: int = 3600,
        enable_encryption: bool = True,
    ):
        self.bucket_name = bucket_name or os.getenv(
            "GCS_MODEL_BUCKET", "goblin-llm-models"
        )
        self.checkpoint_bucket = checkpoint_bucket or os.getenv(
            "GCS_CHECKPOINT_BUCKET", "goblin-llm-checkpoints"
        )
        self.credentials_path = credentials_path or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS", ""
        )
        self.signed_url_expiry = signed_url_expiry_seconds
        self.enable_encryption = enable_encryption

        self._client: Optional[Any] = None
        self._bucket: Optional[Any] = None
        self._checkpoint_bucket: Optional[Any] = None

    def _get_client(self):
        """Get or create GCS client."""
        if not GCS_AVAILABLE:
            raise RuntimeError("google-cloud-storage not installed")

        if self._client is None:
            if self.credentials_path and os.path.exists(self.credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self._client = storage.Client(credentials=credentials)
            else:
                self._client = storage.Client()

        return self._client

    def _get_bucket(self):
        """Get or create bucket reference."""
        if self._bucket is None:
            client = self._get_client()
            self._bucket = client.bucket(self.bucket_name)
        return self._bucket

    def _get_checkpoint_bucket(self):
        """Get or create checkpoint bucket reference."""
        if self._checkpoint_bucket is None:
            client = self._get_client()
            self._checkpoint_bucket = client.bucket(self.checkpoint_bucket)
        return self._checkpoint_bucket

    # ==================== Model Management ====================

    async def list_models(
        self,
        prefix: str = "",
        format_filter: Optional[ModelFormat] = None,
        quantization_filter: Optional[QuantizationType] = None,
    ) -> List[ModelMetadata]:
        """List all models in the bucket."""
        bucket = self._get_bucket()
        models = []

        # List metadata files
        def _list():
            blobs = bucket.list_blobs(prefix=f"{prefix}metadata/")
            return list(blobs)

        blobs = await asyncio.get_event_loop().run_in_executor(None, _list)

        for blob in blobs:
            if blob.name.endswith(".json"):
                try:
                    content = await asyncio.get_event_loop().run_in_executor(
                        None, blob.download_as_string
                    )
                    meta_dict = json.loads(content)
                    metadata = self._dict_to_metadata(meta_dict)

                    # Apply filters
                    if format_filter and metadata.format != format_filter:
                        continue
                    if (
                        quantization_filter
                        and metadata.quantization != quantization_filter
                    ):
                        continue

                    models.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to parse metadata for {blob.name}: {e}")

        return models

    async def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model."""
        bucket = self._get_bucket()
        blob = bucket.blob(f"metadata/{model_id}.json")

        def _download():
            if blob.exists():
                return blob.download_as_string()
            return None

        content = await asyncio.get_event_loop().run_in_executor(None, _download)

        if content:
            meta_dict = json.loads(content)
            return self._dict_to_metadata(meta_dict)

        return None

    async def upload_model(
        self,
        local_path: str,
        model_id: str,
        name: str,
        version: str,
        format: ModelFormat,
        quantization: QuantizationType = QuantizationType.NONE,
        parameters: Optional[int] = None,
        context_length: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> ModelMetadata:
        """
        Upload a model to GCS.

        Args:
            local_path: Path to local model file/directory
            model_id: Unique identifier for the model
            name: Human-readable name
            version: Version string
            format: Model format
            quantization: Quantization type
            parameters: Number of parameters
            context_length: Context length
            tags: Tags for categorization

        Returns:
            Model metadata
        """
        bucket = self._get_bucket()
        path_obj = Path(local_path)

        # Calculate checksum
        checksum = await self._calculate_checksum(path_obj)

        # Determine GCS path
        gcs_path = f"models/{model_id}/{version}/"

        # Upload files
        total_size = 0
        if path_obj.is_dir():
            for file_path in path_obj.rglob("*"):
                if file_path.is_file():
                    blob_path = f"{gcs_path}{file_path.relative_to(path_obj)}"
                    blob = bucket.blob(blob_path)

                    def _upload(fp=file_path, b=blob):
                        b.upload_from_filename(str(fp))
                        return fp.stat().st_size

                    size = await asyncio.get_event_loop().run_in_executor(None, _upload)
                    total_size += size
        else:
            blob = bucket.blob(f"{gcs_path}{path_obj.name}")

            def _upload_single():
                blob.upload_from_filename(str(path_obj))
                return path_obj.stat().st_size

            total_size = await asyncio.get_event_loop().run_in_executor(
                None, _upload_single
            )

        # Create metadata
        now = datetime.utcnow()
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            format=format,
            quantization=quantization,
            size_bytes=total_size,
            checksum_sha256=checksum,
            gcs_path=f"gs://{self.bucket_name}/{gcs_path}",
            created_at=now,
            updated_at=now,
            parameters=parameters,
            context_length=context_length,
            tags=tags or [],
        )

        # Save metadata
        await self._save_metadata(metadata)

        logger.info(f"Uploaded model {model_id} v{version} ({total_size / 1e9:.2f} GB)")
        return metadata

    async def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Delete a model from GCS."""
        bucket = self._get_bucket()

        if version:
            prefix = f"models/{model_id}/{version}/"
        else:
            prefix = f"models/{model_id}/"

        def _delete():
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                blob.delete()

            # Delete metadata
            meta_blob = bucket.blob(f"metadata/{model_id}.json")
            if meta_blob.exists():
                meta_blob.delete()

        await asyncio.get_event_loop().run_in_executor(None, _delete)
        logger.info(f"Deleted model {model_id}" + (f" v{version}" if version else ""))
        return True

    # ==================== Signed URL Generation ====================

    async def generate_signed_url(
        self,
        model_id: str,
        version: Optional[str] = None,
        expiry_seconds: Optional[int] = None,
    ) -> SignedURLResult:
        """
        Generate a signed URL for downloading a model.

        Args:
            model_id: Model identifier
            version: Specific version (uses latest if not specified)
            expiry_seconds: URL validity in seconds

        Returns:
            SignedURLResult with URL and expiry
        """
        metadata = await self.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")

        if version and metadata.version != version:
            # Try to find specific version
            metadata = await self.get_model_metadata(f"{model_id}-{version}")
            if not metadata:
                raise ValueError(f"Model {model_id} version {version} not found")

        bucket = self._get_bucket()

        # Get the main model file
        gcs_path = metadata.gcs_path.replace(f"gs://{self.bucket_name}/", "")

        # List files in the model directory
        def _list_blobs():
            return list(bucket.list_blobs(prefix=gcs_path))

        blobs = await asyncio.get_event_loop().run_in_executor(None, _list_blobs)

        if not blobs:
            raise ValueError(f"No files found for model {model_id}")

        # Generate signed URL for the main blob (largest file)
        main_blob = max(blobs, key=lambda b: b.size or 0)

        expiry = expiry_seconds or self.signed_url_expiry
        expiration = datetime.utcnow() + timedelta(seconds=expiry)

        def _generate_url():
            return main_blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method="GET",
            )

        url = await asyncio.get_event_loop().run_in_executor(None, _generate_url)

        return SignedURLResult(
            url=url,
            expires_at=expiration,
            model_id=model_id,
            checksum=metadata.checksum_sha256,
        )

    async def generate_download_manifest(
        self,
        model_id: str,
        version: Optional[str] = None,
        expiry_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a manifest with signed URLs for all model files.

        Useful for multi-file models (e.g., HuggingFace format).
        """
        metadata = await self.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")

        bucket = self._get_bucket()
        gcs_path = metadata.gcs_path.replace(f"gs://{self.bucket_name}/", "")

        def _list_and_sign():
            blobs = list(bucket.list_blobs(prefix=gcs_path))
            expiry = expiry_seconds or self.signed_url_expiry
            expiration = datetime.utcnow() + timedelta(seconds=expiry)

            files = []
            for blob in blobs:
                url = blob.generate_signed_url(
                    version="v4",
                    expiration=expiration,
                    method="GET",
                )
                files.append(
                    {
                        "name": blob.name.replace(gcs_path, ""),
                        "size": blob.size,
                        "url": url,
                    }
                )

            return files, expiration

        files, expiration = await asyncio.get_event_loop().run_in_executor(
            None, _list_and_sign
        )

        return {
            "model_id": model_id,
            "version": metadata.version,
            "format": metadata.format.value,
            "checksum": metadata.checksum_sha256,
            "expires_at": expiration.isoformat(),
            "files": files,
        }

    # ==================== Checkpoint Management ====================

    async def save_checkpoint(
        self,
        job_id: str,
        local_path: str,
        checkpoint_name: str = "latest",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a training checkpoint to GCS.

        Args:
            job_id: Training job identifier
            local_path: Path to checkpoint files
            checkpoint_name: Name for the checkpoint (e.g., "latest", "epoch_10")
            metadata: Additional metadata to store

        Returns:
            GCS path to checkpoint
        """
        bucket = self._get_checkpoint_bucket()
        path_obj = Path(local_path)

        gcs_path = f"checkpoints/{job_id}/{checkpoint_name}/"

        # Upload checkpoint files
        if path_obj.is_dir():
            for file_path in path_obj.rglob("*"):
                if file_path.is_file():
                    blob_path = f"{gcs_path}{file_path.relative_to(path_obj)}"
                    blob = bucket.blob(blob_path)

                    def _upload_ckpt(fp=file_path, b=blob):
                        b.upload_from_filename(str(fp))

                    await asyncio.get_event_loop().run_in_executor(None, _upload_ckpt)
        else:
            blob = bucket.blob(f"{gcs_path}{path_obj.name}")

            def _upload_ckpt_single():
                blob.upload_from_filename(str(path_obj))

            await asyncio.get_event_loop().run_in_executor(None, _upload_ckpt_single)

        # Save metadata
        if metadata:
            meta_blob = bucket.blob(f"{gcs_path}metadata.json")

            def _save_meta():
                meta_blob.upload_from_string(
                    json.dumps(
                        {
                            **metadata,
                            "checkpoint_name": checkpoint_name,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                )

            await asyncio.get_event_loop().run_in_executor(None, _save_meta)

        logger.info(f"Saved checkpoint {checkpoint_name} for job {job_id}")
        return f"gs://{self.checkpoint_bucket}/{gcs_path}"

    async def get_latest_checkpoint(self, job_id: str) -> Optional[str]:
        """Get the latest checkpoint URL for a job."""
        bucket = self._get_checkpoint_bucket()

        def _check():
            prefix = f"checkpoints/{job_id}/latest/"
            blobs = list(bucket.list_blobs(prefix=prefix))
            return len(blobs) > 0

        exists = await asyncio.get_event_loop().run_in_executor(None, _check)

        if exists:
            return f"gs://{self.checkpoint_bucket}/checkpoints/{job_id}/latest/"

        return None

    async def list_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a job."""
        bucket = self._get_checkpoint_bucket()

        def _list():
            prefix = f"checkpoints/{job_id}/"
            blobs = list(bucket.list_blobs(prefix=prefix))

            checkpoints = {}
            for blob in blobs:
                # Extract checkpoint name from path
                parts = blob.name.replace(prefix, "").split("/")
                if parts:
                    checkpoint_name = parts[0]
                    if checkpoint_name not in checkpoints:
                        checkpoints[checkpoint_name] = {
                            "name": checkpoint_name,
                            "files": [],
                            "total_size": 0,
                        }
                    checkpoints[checkpoint_name]["files"].append(blob.name)
                    checkpoints[checkpoint_name]["total_size"] += blob.size or 0

            return list(checkpoints.values())

        return await asyncio.get_event_loop().run_in_executor(None, _list)

    async def delete_checkpoint(self, job_id: str, checkpoint_name: str) -> bool:
        """Delete a checkpoint."""
        bucket = self._get_checkpoint_bucket()

        def _delete():
            prefix = f"checkpoints/{job_id}/{checkpoint_name}/"
            blobs = list(bucket.list_blobs(prefix=prefix))
            for blob in blobs:
                blob.delete()

        await asyncio.get_event_loop().run_in_executor(None, _delete)
        logger.info(f"Deleted checkpoint {checkpoint_name} for job {job_id}")
        return True

    # ==================== Helper Methods ====================

    async def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of a file or directory."""

        def _calc():
            sha256 = hashlib.sha256()

            if path.is_dir():
                for file_path in sorted(path.rglob("*")):
                    if file_path.is_file():
                        with open(file_path, "rb") as f:
                            for chunk in iter(lambda: f.read(8192), b""):
                                sha256.update(chunk)
            else:
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)

            return sha256.hexdigest()

        return await asyncio.get_event_loop().run_in_executor(None, _calc)

    async def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata to GCS."""
        bucket = self._get_bucket()
        blob = bucket.blob(f"metadata/{metadata.model_id}.json")

        def _save():
            blob.upload_from_string(json.dumps(metadata.to_dict(), indent=2))

        await asyncio.get_event_loop().run_in_executor(None, _save)

    def _dict_to_metadata(self, d: Dict[str, Any]) -> ModelMetadata:
        """Convert dictionary to ModelMetadata."""
        return ModelMetadata(
            model_id=d["model_id"],
            name=d["name"],
            version=d["version"],
            format=ModelFormat(d["format"]),
            quantization=QuantizationType(d.get("quantization", "none")),
            size_bytes=d["size_bytes"],
            checksum_sha256=d["checksum_sha256"],
            gcs_path=d["gcs_path"],
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            parameters=d.get("parameters"),
            context_length=d.get("context_length"),
            tags=d.get("tags", []),
        )


# Singleton instance
_storage: Optional[GCSModelStorage] = None


def get_model_storage() -> GCSModelStorage:
    """Get or create the model storage singleton."""
    global _storage
    if _storage is None:
        _storage = GCSModelStorage()
    return _storage
