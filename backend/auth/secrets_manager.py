"""
Secrets management using Bitwarden CLI or HashiCorp Vault.

Provides a unified interface for retrieving secrets from various sources.
Prioritizes Bitwarden for development and Vault for production.
"""

import os
import json
import subprocess
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecretsProvider(Enum):
    """Available secrets providers."""

    BITWARDEN = "bitwarden"
    VAULT = "vault"
    ENVIRONMENT = "environment"


@dataclass
class SecretConfig:
    """Configuration for secrets management."""

    provider: SecretsProvider
    vault_url: Optional[str] = None
    vault_token: Optional[str] = None
    bitwarden_session: Optional[str] = None
    bitwarden_server_url: Optional[str] = None


class SecretsManager:
    """Unified secrets management interface."""

    def __init__(self, config: Optional[SecretConfig] = None):
        """Initialize secrets manager.

        Args:
            config: Secrets configuration. If None, auto-detects from environment.
        """
        self.config = config or self._auto_detect_config()
        self._cache: Dict[str, Any] = {}

    def _auto_detect_config(self) -> SecretConfig:
        """Auto-detect secrets configuration from environment."""
        # Check for Vault first (production)
        vault_url = os.getenv("VAULT_ADDR")
        vault_token = os.getenv("VAULT_TOKEN")

        if vault_url and vault_token:
            return SecretConfig(
                provider=SecretsProvider.VAULT,
                vault_url=vault_url,
                vault_token=vault_token,
            )

        # Check for Bitwarden (development)
        bw_session = os.getenv("BW_SESSION")
        bw_server = os.getenv("BW_SERVER_URL")

        if bw_session:
            return SecretConfig(
                provider=SecretsProvider.BITWARDEN,
                bitwarden_session=bw_session,
                bitwarden_server_url=bw_server,
            )

        # Fallback to environment variables
        return SecretConfig(provider=SecretsProvider.ENVIRONMENT)

    def get_secret(self, key: str, default: Any = None) -> Any:
        """Get a secret value.

        Args:
            key: Secret key/name
            default: Default value if secret not found

        Returns:
            Secret value or default
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        try:
            if self.config.provider == SecretsProvider.VAULT:
                value = self._get_from_vault(key)
            elif self.config.provider == SecretsProvider.BITWARDEN:
                value = self._get_from_bitwarden(key)
            else:
                value = self._get_from_environment(key)

            # Only cache if we got a non-None value
            if value is not None:
                self._cache[key] = value

            return value if value is not None else default

        except Exception as e:
            logger.warning(f"Failed to retrieve secret '{key}': {e}")
            return default

    def _get_from_vault(self, key: str) -> Any:
        """Get secret from HashiCorp Vault."""
        if not self.config.vault_url or not self.config.vault_token:
            raise ValueError("Vault URL and token required")

        # Parse key as path/to/secret:field
        if ":" in key:
            path, field = key.split(":", 1)
        else:
            path, field = key, None

        url = f"{self.config.vault_url}/v1/{path}"
        headers = {
            "X-Vault-Token": self.config.vault_token,
            "Content-Type": "application/json",
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()["data"]

        if field:
            return data.get(field)
        return data

    def _get_from_bitwarden(self, key: str) -> Any:
        """Get secret from Bitwarden CLI."""
        try:
            # Use bw get to retrieve the item
            cmd = ["bw", "get", "item", key]

            if self.config.bitwarden_server_url:
                cmd.extend(["--server", self.config.bitwarden_server_url])

            env = os.environ.copy()
            if self.config.bitwarden_session:
                env["BW_SESSION"] = self.config.bitwarden_session

            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=30
            )

            if result.returncode != 0:
                raise ValueError(f"Bitwarden CLI error: {result.stderr}")

            # Parse JSON output
            item = json.loads(result.stdout)

            # Return the password field by default, or the whole item
            if "login" in item and item["login"] and "password" in item["login"]:
                return item["login"]["password"]
            else:
                return item

        except subprocess.TimeoutExpired:
            raise ValueError("Bitwarden CLI timeout")
        except json.JSONDecodeError:
            raise ValueError("Invalid Bitwarden item format")

    def _get_from_environment(self, key: str) -> Any:
        """Get secret from environment variables."""
        return os.getenv(key)

    def get_jwt_secret_key(self) -> str:
        """Get JWT secret key.

        Raises:
            RuntimeError: If JWT_SECRET_KEY is not configured in the environment
                or secrets provider. A secure random key must be set explicitly;
                hardcoded fallbacks are not allowed.
        """
        value = self.get_secret("JWT_SECRET_KEY")
        if not value:
            raise RuntimeError(
                "JWT_SECRET_KEY is not configured. Set it via environment variable, "
                "Vault, or Bitwarden before starting the server. "
                "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        return value

    def get_api_key_store_encryption_key(self) -> Optional[str]:
        """Get API key store encryption key."""
        return self.get_secret("API_KEY_STORE_ENCRYPTION_KEY")

    def get_database_url(self) -> Optional[str]:
        """Get database connection URL."""
        return self.get_secret("DATABASE_URL")

    def get_supabase_keys(self) -> Dict[str, str]:
        """Get Supabase configuration."""
        return {
            "url": self.get_secret("SUPABASE_URL"),
            "anon_key": self.get_secret("SUPABASE_ANON_KEY"),
            "service_key": self.get_secret("SUPABASE_SERVICE_ROLE_KEY"),
        }

    def get_provider_api_keys(self) -> Dict[str, str]:
        """Get provider API keys."""
        keys = {}

        # Common providers
        providers = ["openai", "anthropic", "groq", "together", "fireworks", "ollama"]

        for provider in providers:
            key = self.get_secret(f"{provider.upper()}_API_KEY")
            if key:
                keys[provider] = key

        return keys

    def invalidate_cache(self, key: Optional[str] = None):
        """Invalidate cached secrets.

        Args:
            key: Specific key to invalidate, or None for all
        """
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()


# Global instance
secrets_manager = SecretsManager()


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    return secrets_manager
