"""
Model Registry Service - Hybrid Strategy

Tracks experiments & metrics in MLflow/Weights & Biases while storing
artifacts/weights in GCS with consistent naming.

Architecture:
- Registry (MLflow/W&B) = Authoritative source for deployment + rollback
- GCS = Artifact storage (cheap, reliable, single-region for alpha)

Naming Convention:
    gs://{bucket}/models/{org}/{model_name}/{version}/{artifact}

    Examples:
    gs://goblin-llm-models/models/goblin/mistral-7b-instruct/v1.0.0/model.safetensors
    gs://goblin-llm-models/models/goblin/mistral-7b-instruct/v1.0.0/config.json
    gs://goblin-llm-models/models/goblin/mistral-7b-instruct/v1.0.0/metadata.json
"""

import os
import json
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional MLflow/W&B imports
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.info("MLflow not installed. Install with: pip install mlflow")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.info("Weights & Biases not installed. Install with: pip install wandb")


class ModelStage(Enum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ExperimentTracker(Enum):
    """Supported experiment tracking backends."""

    MLFLOW = "mlflow"
    WANDB = "wandb"
    BOTH = "both"
    NONE = "none"


@dataclass
class ModelVersion:
    """Represents a registered model version."""

    model_name: str
    version: str
    stage: ModelStage
    gcs_uri: str
    run_id: Optional[str] = None  # MLflow/W&B run ID
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""

    # Artifact info
    checksum_sha256: str = ""
    size_bytes: int = 0
    format: str = "safetensors"  # pytorch, safetensors, gguf, onnx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "stage": self.stage.value,
            "gcs_uri": self.gcs_uri,
            "run_id": self.run_id,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "checksum_sha256": self.checksum_sha256,
            "size_bytes": self.size_bytes,
            "format": self.format,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        return cls(
            model_name=data["model_name"],
            version=data["version"],
            stage=ModelStage(data["stage"]),
            gcs_uri=data["gcs_uri"],
            run_id=data.get("run_id"),
            metrics=data.get("metrics", {}),
            parameters=data.get("parameters", {}),
            tags=data.get("tags", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data.get("description", ""),
            checksum_sha256=data.get("checksum_sha256", ""),
            size_bytes=data.get("size_bytes", 0),
            format=data.get("format", "safetensors"),
        )


@dataclass
class RegistryConfig:
    """Configuration for the model registry."""

    # GCS storage (single region for alpha - us-east1)
    gcs_bucket: str = os.getenv("GCS_MODEL_BUCKET", "goblin-llm-models")
    gcs_region: str = os.getenv("GCS_REGION", "us-east1")  # Single region for alpha
    organization: str = os.getenv("REGISTRY_ORG", "goblin")

    # Experiment tracking
    tracker: ExperimentTracker = ExperimentTracker.MLFLOW
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_registry_uri: str = os.getenv("MLFLOW_REGISTRY_URI", "")
    wandb_project: str = os.getenv("WANDB_PROJECT", "goblin-llm")
    wandb_entity: str = os.getenv("WANDB_ENTITY", "")

    # Local registry cache
    local_cache_dir: str = os.getenv("REGISTRY_CACHE_DIR", "/tmp/goblin-registry-cache")

    # Versioning
    auto_version: bool = True  # Auto-increment versions
    version_prefix: str = "v"


class ModelRegistry:
    """
    Hybrid Model Registry with MLflow/W&B tracking and GCS storage.

    Usage:
        registry = ModelRegistry()

        # Register a new model version
        version = registry.register_model(
            model_name="mistral-7b-instruct",
            model_path="/path/to/model",
            metrics={"loss": 0.5, "accuracy": 0.95},
            parameters={"epochs": 10, "lr": 1e-5},
        )

        # Get production model
        prod_model = registry.get_production_model("mistral-7b-instruct")

        # Rollback to previous version
        registry.rollback("mistral-7b-instruct", "v1.0.0")
    """

    def __init__(self, config: Optional[RegistryConfig] = None):
        self.config = config or RegistryConfig()
        self._init_tracking()
        self._init_storage()

    def _init_tracking(self):
        """Initialize experiment tracking backend."""
        tracker = self.config.tracker

        if tracker in (ExperimentTracker.MLFLOW, ExperimentTracker.BOTH):
            if MLFLOW_AVAILABLE:
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                if self.config.mlflow_registry_uri:
                    mlflow.set_registry_uri(self.config.mlflow_registry_uri)
                self.mlflow_client = MlflowClient()
                logger.info(
                    f"MLflow tracking initialized: {self.config.mlflow_tracking_uri}"
                )
            else:
                logger.warning("MLflow requested but not available")

        if tracker in (ExperimentTracker.WANDB, ExperimentTracker.BOTH):
            if WANDB_AVAILABLE:
                wandb.login()
                logger.info(f"W&B tracking initialized: {self.config.wandb_project}")
            else:
                logger.warning("W&B requested but not available")

    def _init_storage(self):
        """Initialize GCS storage client."""
        try:
            from google.cloud import storage

            self.gcs_client = storage.Client()
            self.bucket = self.gcs_client.bucket(self.config.gcs_bucket)
            logger.info(f"GCS storage initialized: {self.config.gcs_bucket}")
        except ImportError:
            logger.warning("google-cloud-storage not installed. GCS disabled.")
            self.gcs_client = None
            self.bucket = None
        except Exception as e:
            logger.error(f"Failed to initialize GCS: {e}")
            self.gcs_client = None
            self.bucket = None

    def _generate_gcs_uri(self, model_name: str, version: str) -> str:
        """Generate consistent GCS URI for model artifacts."""
        return (
            f"gs://{self.config.gcs_bucket}/models/"
            f"{self.config.organization}/{model_name}/{version}"
        )

    def _get_next_version(self, model_name: str) -> str:
        """Get next version number for a model."""
        versions = self.list_versions(model_name)
        if not versions:
            return f"{self.config.version_prefix}1.0.0"

        # Parse existing versions and find max
        max_major, max_minor, max_patch = 0, 0, 0
        for v in versions:
            try:
                ver = v.version.lstrip(self.config.version_prefix)
                parts = ver.split(".")
                major = int(parts[0]) if len(parts) > 0 else 0
                minor = int(parts[1]) if len(parts) > 1 else 0
                patch = int(parts[2]) if len(parts) > 2 else 0

                if (major, minor, patch) > (max_major, max_minor, max_patch):
                    max_major, max_minor, max_patch = major, minor, patch
            except (ValueError, IndexError):
                continue

        # Increment patch version
        return f"{self.config.version_prefix}{max_major}.{max_minor}.{max_patch + 1}"

    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        version: Optional[str] = None,
        description: str = "",
        stage: ModelStage = ModelStage.DEVELOPMENT,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_name: Name of the model (e.g., "mistral-7b-instruct")
            model_path: Local path to model artifacts
            metrics: Training/eval metrics to log
            parameters: Training parameters to log
            tags: Additional tags
            version: Explicit version (auto-generated if None)
            description: Model description
            stage: Initial lifecycle stage

        Returns:
            ModelVersion object with registration details
        """
        metrics = metrics or {}
        parameters = parameters or {}
        tags = tags or {}

        # Generate version if not provided
        if version is None and self.config.auto_version:
            version = self._get_next_version(model_name)
        elif version is None:
            version = f"{self.config.version_prefix}1.0.0"

        gcs_uri = self._generate_gcs_uri(model_name, version)
        run_id = None

        # Start experiment tracking
        if self.config.tracker in (ExperimentTracker.MLFLOW, ExperimentTracker.BOTH):
            if MLFLOW_AVAILABLE:
                run_id = self._log_to_mlflow(
                    model_name, version, model_path, metrics, parameters, tags
                )

        if self.config.tracker in (ExperimentTracker.WANDB, ExperimentTracker.BOTH):
            if WANDB_AVAILABLE:
                self._log_to_wandb(
                    model_name, version, model_path, metrics, parameters, tags
                )

        # Upload to GCS
        checksum, size = self._upload_to_gcs(model_name, version, model_path)

        # Create version record
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            stage=stage,
            gcs_uri=gcs_uri,
            run_id=run_id,
            metrics=metrics,
            parameters=parameters,
            tags=tags,
            description=description,
            checksum_sha256=checksum,
            size_bytes=size,
        )

        # Save metadata to GCS
        self._save_metadata(model_version)

        logger.info(f"Registered model: {model_name} {version} at {gcs_uri}")
        return model_version

    def _log_to_mlflow(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        tags: Dict[str, str],
    ) -> Optional[str]:
        """Log experiment to MLflow."""
        try:
            with mlflow.start_run(run_name=f"{model_name}-{version}") as run:
                # Log parameters
                for key, value in parameters.items():
                    mlflow.log_param(key, value)

                # Log metrics
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                # Log tags
                mlflow.set_tags(tags)
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("version", version)

                # Log artifacts (optional - we primarily use GCS)
                if os.path.isfile(model_path):
                    mlflow.log_artifact(model_path)
                elif os.path.isdir(model_path):
                    mlflow.log_artifacts(model_path)

                return run.info.run_id
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")
            return None

    def _log_to_wandb(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        tags: Dict[str, str],
    ):
        """Log experiment to Weights & Biases."""
        try:
            run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity or None,
                name=f"{model_name}-{version}",
                config=parameters,
                tags=list(tags.keys()),
            )

            # Log metrics
            wandb.log(metrics)

            # Log model artifact
            artifact = wandb.Artifact(
                name=f"{model_name}-{version}",
                type="model",
                metadata={"version": version, **tags},
            )
            if os.path.isfile(model_path):
                artifact.add_file(model_path)
            elif os.path.isdir(model_path):
                artifact.add_dir(model_path)
            run.log_artifact(artifact)

            wandb.finish()
        except Exception as e:
            logger.error(f"Failed to log to W&B: {e}")

    def _upload_to_gcs(
        self, model_name: str, version: str, model_path: str
    ) -> tuple[str, int]:
        """Upload model artifacts to GCS."""
        if not self.bucket:
            logger.warning("GCS not available, skipping upload")
            return "", 0

        prefix = f"models/{self.config.organization}/{model_name}/{version}"
        total_size = 0
        combined_hash = hashlib.sha256()

        path = Path(model_path)

        if path.is_file():
            # Single file upload
            blob_name = f"{prefix}/{path.name}"
            blob = self.bucket.blob(blob_name)

            with open(path, "rb") as f:
                content = f.read()
                combined_hash.update(content)
                total_size = len(content)

            blob.upload_from_filename(str(path))
            logger.info(f"Uploaded: {blob_name}")

        elif path.is_dir():
            # Directory upload
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(path)
                    blob_name = f"{prefix}/{rel_path}"
                    blob = self.bucket.blob(blob_name)

                    with open(file_path, "rb") as f:
                        content = f.read()
                        combined_hash.update(content)
                        total_size += len(content)

                    blob.upload_from_filename(str(file_path))
                    logger.debug(f"Uploaded: {blob_name}")

            logger.info(
                f"Uploaded directory to: gs://{self.config.gcs_bucket}/{prefix}"
            )

        return combined_hash.hexdigest(), total_size

    def _save_metadata(self, version: ModelVersion):
        """Save model metadata to GCS."""
        if not self.bucket:
            return

        prefix = (
            f"models/{self.config.organization}/{version.model_name}/{version.version}"
        )
        blob = self.bucket.blob(f"{prefix}/metadata.json")
        blob.upload_from_string(
            json.dumps(version.to_dict(), indent=2),
            content_type="application/json",
        )

    def get_model(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        if not self.bucket:
            return None

        prefix = f"models/{self.config.organization}/{model_name}/{version}"
        blob = self.bucket.blob(f"{prefix}/metadata.json")

        try:
            content = blob.download_as_text()
            return ModelVersion.from_dict(json.loads(content))
        except Exception as e:
            logger.error(f"Failed to get model {model_name}:{version}: {e}")
            return None

    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get the current production model version."""
        versions = self.list_versions(model_name)
        for v in versions:
            if v.stage == ModelStage.PRODUCTION:
                return v
        return None

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        if not self.bucket:
            return []

        prefix = f"models/{self.config.organization}/{model_name}/"
        versions = []

        # List all version folders
        blobs = self.gcs_client.list_blobs(
            self.config.gcs_bucket, prefix=prefix, delimiter="/"
        )

        # Need to iterate to get prefixes
        list(blobs)  # Consume iterator
        for blob_prefix in blobs.prefixes:
            version_name = blob_prefix.rstrip("/").split("/")[-1]
            metadata_blob = self.bucket.blob(f"{blob_prefix}metadata.json")

            try:
                content = metadata_blob.download_as_text()
                versions.append(ModelVersion.from_dict(json.loads(content)))
            except Exception:
                # Create minimal version from path
                versions.append(
                    ModelVersion(
                        model_name=model_name,
                        version=version_name,
                        stage=ModelStage.DEVELOPMENT,
                        gcs_uri=f"gs://{self.config.gcs_bucket}/{blob_prefix}",
                    )
                )

        # Sort by version (newest first)
        versions.sort(key=lambda v: v.version, reverse=True)
        return versions

    def promote_to_staging(self, model_name: str, version: str) -> bool:
        """Promote a model version to staging."""
        return self._update_stage(model_name, version, ModelStage.STAGING)

    def promote_to_production(self, model_name: str, version: str) -> bool:
        """Promote a model version to production."""
        # First, demote current production
        current_prod = self.get_production_model(model_name)
        if current_prod:
            self._update_stage(model_name, current_prod.version, ModelStage.ARCHIVED)

        return self._update_stage(model_name, version, ModelStage.PRODUCTION)

    def rollback(self, model_name: str, target_version: str) -> bool:
        """
        Rollback production to a specific version.

        Args:
            model_name: Name of the model
            target_version: Version to rollback to

        Returns:
            True if successful
        """
        logger.info(f"Rolling back {model_name} to {target_version}")

        # Demote current production
        current_prod = self.get_production_model(model_name)
        if current_prod:
            self._update_stage(model_name, current_prod.version, ModelStage.ARCHIVED)

        # Promote target to production
        return self._update_stage(model_name, target_version, ModelStage.PRODUCTION)

    def _update_stage(self, model_name: str, version: str, stage: ModelStage) -> bool:
        """Update the stage of a model version."""
        model_version = self.get_model(model_name, version)
        if not model_version:
            logger.error(f"Model not found: {model_name}:{version}")
            return False

        model_version.stage = stage
        self._save_metadata(model_version)

        logger.info(f"Updated {model_name}:{version} to stage {stage.value}")
        return True

    def get_download_url(
        self, model_name: str, version: str, artifact: str = "", expiry_hours: int = 1
    ) -> Optional[str]:
        """
        Get a signed download URL for model artifacts.

        Args:
            model_name: Model name
            version: Model version
            artifact: Specific artifact path (empty for entire model dir)
            expiry_hours: URL expiry time

        Returns:
            Signed URL string
        """
        if not self.bucket:
            return None

        from datetime import timedelta

        prefix = f"models/{self.config.organization}/{model_name}/{version}"
        if artifact:
            prefix = f"{prefix}/{artifact}"

        blob = self.bucket.blob(prefix)

        try:
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=expiry_hours),
                method="GET",
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed URL: {e}")
            return None


# Singleton instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
