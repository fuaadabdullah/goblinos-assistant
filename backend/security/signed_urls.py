"""
Signed URL Generator
Short-lived URLs for secure weight access from GCS
"""

from datetime import datetime, timedelta
from typing import Optional

from google.cloud import storage
from google.auth import default
import structlog

from ..config import settings

logger = structlog.get_logger()


class SignedURLGenerator:
    """
    Generate short-lived signed URLs for GCS objects.
    
    Used to:
    - Pull weights at container start
    - Avoid baking credentials into images
    - Limit exposure window (15-minute default)
    """
    
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        expiry_minutes: Optional[int] = None,
    ):
        self.bucket_name = bucket_name or settings.gcs_weights_bucket
        self.expiry_minutes = expiry_minutes or settings.signed_url_expiry_minutes
        self._client: Optional[storage.Client] = None
    
    @property
    def client(self) -> storage.Client:
        """Lazy-load storage client."""
        if self._client is None:
            self._client = storage.Client(project=settings.gcp_project_id)
        return self._client
    
    def generate_download_url(
        self,
        blob_path: str,
        expiry_minutes: Optional[int] = None,
        bucket_name: Optional[str] = None,
    ) -> str:
        """
        Generate a signed URL for downloading a blob.
        
        Args:
            blob_path: Path to blob in bucket
            expiry_minutes: URL validity (default from settings)
            bucket_name: Override bucket name
            
        Returns:
            Signed URL string
        """
        bucket = self.client.bucket(bucket_name or self.bucket_name)
        blob = bucket.blob(blob_path)
        
        expiry = timedelta(minutes=expiry_minutes or self.expiry_minutes)
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=expiry,
            method="GET",
        )
        
        logger.info(
            "Generated download URL",
            bucket=bucket.name,
            blob=blob_path,
            expiry_minutes=expiry_minutes or self.expiry_minutes,
        )
        
        return url
    
    def generate_upload_url(
        self,
        blob_path: str,
        expiry_minutes: Optional[int] = None,
        bucket_name: Optional[str] = None,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Generate a signed URL for uploading a blob.
        
        Args:
            blob_path: Target path in bucket
            expiry_minutes: URL validity
            bucket_name: Override bucket name
            content_type: Expected content type
            
        Returns:
            Signed URL string
        """
        bucket = self.client.bucket(bucket_name or self.bucket_name)
        blob = bucket.blob(blob_path)
        
        expiry = timedelta(minutes=expiry_minutes or self.expiry_minutes)
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=expiry,
            method="PUT",
            content_type=content_type,
        )
        
        logger.info(
            "Generated upload URL",
            bucket=bucket.name,
            blob=blob_path,
            expiry_minutes=expiry_minutes or self.expiry_minutes,
        )
        
        return url
    
    def generate_model_urls(
        self,
        model_name: str,
        files: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """
        Generate signed URLs for all files of a model.
        
        Args:
            model_name: Model directory name in bucket
            files: Specific files to generate URLs for
                   If None, generates for common model files
                   
        Returns:
            Dict mapping filename to signed URL
        """
        common_files = files or [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
        ]
        
        urls = {}
        bucket = self.client.bucket(self.bucket_name)
        
        for filename in common_files:
            blob_path = f"{model_name}/{filename}"
            blob = bucket.blob(blob_path)
            
            if blob.exists():
                urls[filename] = self.generate_download_url(blob_path)
        
        return urls
    
    def list_models(self) -> list[str]:
        """List all model directories in the weights bucket."""
        bucket = self.client.bucket(self.bucket_name)
        
        # Get unique prefixes (directories)
        models = set()
        for blob in bucket.list_blobs():
            parts = blob.name.split("/")
            if len(parts) > 1:
                models.add(parts[0])
        
        return sorted(models)


# Singleton instance
_url_generator: Optional[SignedURLGenerator] = None


def get_url_generator() -> SignedURLGenerator:
    """Get or create URL generator singleton."""
    global _url_generator
    if _url_generator is None:
        _url_generator = SignedURLGenerator()
    return _url_generator
