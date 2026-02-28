"""
Secrets Manager
Centralized secrets management using Google Secret Manager
"""

from typing import Any, Dict, Optional

from google.cloud import secretmanager
import structlog

from ..config import settings

logger = structlog.get_logger()


class SecretsManager:
    """
    Centralized secrets management.
    
    Uses Google Secret Manager for production,
    with fallback to environment variables for dev.
    """
    
    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or settings.gcp_project_id
        self._client: Optional[secretmanager.SecretManagerServiceClient] = None
        self._cache: Dict[str, str] = {}
    
    @property
    def client(self) -> secretmanager.SecretManagerServiceClient:
        """Lazy-load Secret Manager client."""
        if self._client is None:
            self._client = secretmanager.SecretManagerServiceClient()
        return self._client
    
    def get_secret(
        self,
        secret_id: str,
        version: str = "latest",
        use_cache: bool = True,
    ) -> str:
        """
        Get a secret value.
        
        Args:
            secret_id: Secret identifier
            version: Secret version (default: latest)
            use_cache: Whether to cache the value
            
        Returns:
            Secret value as string
        """
        cache_key = f"{secret_id}:{version}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
            response = self.client.access_secret_version(request={"name": name})
            value = response.payload.data.decode("UTF-8")
            
            if use_cache:
                self._cache[cache_key] = value
            
            logger.debug("Retrieved secret", secret_id=secret_id)
            return value
            
        except Exception as e:
            logger.error("Failed to get secret", secret_id=secret_id, error=str(e))
            raise
    
    def set_secret(
        self,
        secret_id: str,
        value: str,
        create_if_missing: bool = True,
    ) -> str:
        """
        Set a secret value (creates new version).
        
        Args:
            secret_id: Secret identifier
            value: Secret value
            create_if_missing: Create secret if it doesn't exist
            
        Returns:
            Version name of the created secret
        """
        parent = f"projects/{self.project_id}"
        secret_path = f"{parent}/secrets/{secret_id}"
        
        try:
            # Check if secret exists
            try:
                self.client.get_secret(request={"name": secret_path})
            except Exception:
                if create_if_missing:
                    # Create secret
                    self.client.create_secret(
                        request={
                            "parent": parent,
                            "secret_id": secret_id,
                            "secret": {"replication": {"automatic": {}}},
                        }
                    )
                else:
                    raise
            
            # Add new version
            response = self.client.add_secret_version(
                request={
                    "parent": secret_path,
                    "payload": {"data": value.encode("UTF-8")},
                }
            )
            
            # Invalidate cache
            self._cache = {k: v for k, v in self._cache.items() if not k.startswith(secret_id)}
            
            logger.info("Secret updated", secret_id=secret_id, version=response.name)
            return response.name
            
        except Exception as e:
            logger.error("Failed to set secret", secret_id=secret_id, error=str(e))
            raise
    
    def delete_secret(self, secret_id: str) -> bool:
        """
        Delete a secret.
        
        Args:
            secret_id: Secret identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            name = f"projects/{self.project_id}/secrets/{secret_id}"
            self.client.delete_secret(request={"name": name})
            
            # Clear cache
            self._cache = {k: v for k, v in self._cache.items() if not k.startswith(secret_id)}
            
            logger.info("Secret deleted", secret_id=secret_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete secret", secret_id=secret_id, error=str(e))
            return False
    
    def list_secrets(self) -> list[str]:
        """List all secrets in the project."""
        parent = f"projects/{self.project_id}"
        
        secrets = []
        for secret in self.client.list_secrets(request={"parent": parent}):
            # Extract secret ID from name
            secret_id = secret.name.split("/")[-1]
            secrets.append(secret_id)
        
        return secrets
    
    def clear_cache(self) -> None:
        """Clear the secrets cache."""
        self._cache = {}


# Singleton instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get or create secrets manager singleton."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager
