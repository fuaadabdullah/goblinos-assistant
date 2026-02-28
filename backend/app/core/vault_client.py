# Vault Client Implementation
# Goblin Assistant Secrets Management Migration

import os
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import hvac
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


@dataclass
class VaultConfig:
    """Vault configuration settings"""

    address: str
    mount_point: str = "secret"
    auth_method: str = "cert"  # cert, token, oidc
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    token: Optional[str] = None
    role_id: Optional[str] = None
    secret_id: Optional[str] = None


class VaultClient:
    """
    HashiCorp Vault client for secrets management.
    Supports certificate-based authentication for applications.
    """

    def __init__(self, config: VaultConfig):
        self.config = config
        self.client = hvac.Client(url=config.address)

        # Authenticate based on method
        if config.auth_method == "cert" and config.cert_path and config.key_path:
            self._authenticate_cert()
        elif config.auth_method == "token" and config.token:
            self.client.token = config.token
        elif config.auth_method == "approle" and config.role_id and config.secret_id:
            self._authenticate_approle()
        else:
            raise ValueError(f"Unsupported auth method: {config.auth_method}")

        # Verify authentication
        if not self.client.is_authenticated():
            raise RuntimeError("Failed to authenticate with Vault")

        logger.info("Successfully authenticated with Vault")

    def _authenticate_cert(self):
        """Authenticate using client certificates"""
        with open(self.config.cert_path, "r") as cert_file:
            cert = cert_file.read()
        with open(self.config.key_path, "r") as key_file:
            key = key_file.read()

        # Use certificate auth
        self.client.auth.cert.login(cert_pem=cert, key_pem=key)

    def _authenticate_approle(self):
        """Authenticate using AppRole"""
        response = self.client.auth.approle.login(
            role_id=self.config.role_id, secret_id=self.config.secret_id
        )
        self.client.token = response["auth"]["client_token"]

    def get_secret(self, path: str, key: Optional[str] = None) -> Any:
        """
        Retrieve a secret from Vault KV-v2 engine.

        Args:
            path: Secret path (e.g., "production/api-keys/openai")
            key: Specific key within the secret (optional)

        Returns:
            Secret value or dict of all key-value pairs
        """
        try:
            full_path = f"{self.config.mount_point}/data/{path}"
            response = self.client.secrets.kv.v2.read_secret_version(path=full_path)

            data = response["data"]["data"]

            if key:
                return data.get(key)
            return data

        except Exception as e:
            logger.error(f"Failed to retrieve secret from {path}: {e}")
            raise

    def get_database_credentials(self, role: str) -> Dict[str, str]:
        """
        Get dynamic database credentials from Vault.

        Args:
            role: Database role name (e.g., "goblin-assistant-app")

        Returns:
            Dict containing username, password, and lease info
        """
        try:
            response = self.client.secrets.database.generate_credentials(
                name=role, mount_point="database"
            )

            return {
                "username": response["data"]["username"],
                "password": response["data"]["password"],
                "lease_id": response["lease_id"],
                "lease_duration": response["lease_duration"],
            }

        except Exception as e:
            logger.error(f"Failed to get database credentials for role {role}: {e}")
            raise

    def create_secret(self, path: str, data: Dict[str, Any]) -> bool:
        """
        Create or update a secret in Vault KV-v2 engine.

        Args:
            path: Secret path
            data: Key-value pairs to store

        Returns:
            True if successful
        """
        try:
            full_path = f"{self.config.mount_point}/data/{path}"
            self.client.secrets.kv.v2.create_or_update_secret(
                path=full_path, secret=data
            )
            logger.info(f"Successfully created/updated secret at {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create secret at {path}: {e}")
            raise


class MigrationHelper:
    """
    Helper class for migrating from Fernet encryption to Vault.
    Provides backward compatibility during transition.
    """

    def __init__(self, vault_client: VaultClient, fernet_key: Optional[str] = None):
        self.vault_client = vault_client
        self.fernet = None

        if fernet_key:
            # Create Fernet cipher for backward compatibility
            key_bytes = base64.urlsafe_b64decode(fernet_key)
            self.fernet = Fernet(key_bytes)

    def migrate_encrypted_secret(
        self, encrypted_value: str, vault_path: str, vault_key: str
    ) -> str:
        """
        Migrate a Fernet-encrypted secret to Vault and return the plain value.

        Args:
            encrypted_value: Base64-encoded Fernet encrypted value
            vault_path: Vault path to store the secret
            vault_key: Key name in Vault

        Returns:
            Decrypted plain text value
        """
        if not self.fernet:
            raise RuntimeError("Fernet cipher not initialized")

        # Decrypt the value
        decrypted = self.fernet.decrypt(encrypted_value.encode()).decode()

        # Store in Vault
        self.vault_client.create_secret(vault_path, {vault_key: decrypted})

        logger.info(f"Migrated secret from Fernet to Vault at {vault_path}/{vault_key}")
        return decrypted

    def get_secret_with_fallback(
        self, vault_path: str, vault_key: str, encrypted_fallback: Optional[str] = None
    ) -> Optional[str]:
        """
        Get secret from Vault with Fernet fallback for backward compatibility.

        Args:
            vault_path: Vault path
            vault_key: Vault key
            encrypted_fallback: Fernet-encrypted fallback value

        Returns:
            Secret value or None if not found
        """
        try:
            # Try Vault first
            return self.vault_client.get_secret(vault_path, vault_key)
        except Exception as e:
            logger.warning(f"Failed to get secret from Vault: {e}")

            # Fall back to Fernet decryption
            if encrypted_fallback and self.fernet:
                try:
                    decrypted = self.fernet.decrypt(
                        encrypted_fallback.encode()
                    ).decode()
                    logger.info(f"Using Fernet fallback for {vault_path}/{vault_key}")
                    return decrypted
                except Exception as e:
                    logger.error(f"Fernet fallback also failed: {e}")

            return None


# Configuration factory functions
def create_vault_config_from_env() -> VaultConfig:
    """Create Vault configuration from environment variables"""
    return VaultConfig(
        address=os.getenv("VAULT_ADDR", "https://vault.goblin.fuaad.ai"),
        auth_method=os.getenv("VAULT_AUTH_METHOD", "cert"),
        cert_path=os.getenv("VAULT_CLIENT_CERT"),
        key_path=os.getenv("VAULT_CLIENT_KEY"),
        token=os.getenv("VAULT_TOKEN"),
        role_id=os.getenv("VAULT_ROLE_ID"),
        secret_id=os.getenv("VAULT_SECRET_ID"),
    )


def create_vault_client() -> VaultClient:
    """Create and return authenticated Vault client"""
    config = create_vault_config_from_env()
    return VaultClient(config)


# Convenience functions for common operations
def get_api_key(provider: str, environment: str = "production") -> Optional[str]:
    """Get API key for a provider from Vault"""
    client = create_vault_client()
    return client.get_secret(f"{environment}/api-keys/{provider}", "api_key")


def get_database_credentials(role: str = "goblin-assistant-app") -> Dict[str, str]:
    """Get dynamic database credentials"""
    client = create_vault_client()
    return client.get_database_credentials(role)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    try:
        # Create Vault client
        vault = create_vault_client()

        # Get API key
        openai_key = vault.get_secret("production/api-keys/openai", "api_key")
        print(f"OpenAI API Key: {openai_key[:10]}...")

        # Get database credentials
        db_creds = vault.get_database_credentials("goblin-assistant-app")
        print(f"DB Username: {db_creds['username']}")

    except Exception as e:
        logger.error(f"Vault client example failed: {e}")
