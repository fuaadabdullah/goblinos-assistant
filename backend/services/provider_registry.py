"""
Provider adapter registry for managing LLM provider adapters.

This module handles provider adapter initialization, API key management,
and model discovery. Extracted from routing.py for better modularity.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Type

from ..providers.base import ProviderBase
from .encryption import EncryptionService

logger = logging.getLogger(__name__)


def _provider_env_key_candidates(provider_name: str) -> List[str]:
    """Generate possible environment variable names for a provider's API key.

    Args:
        provider_name: Provider name (e.g., "openai", "anthropic")

    Returns:
        List of environment variable names to check
    """
    canonical = (provider_name or "").lower().replace("-", "_")
    if canonical in {"azure_openai", "azure"}:
        return ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY"]
    if canonical in {"aliyun", "alibaba", "alibaba_cloud"}:
        return ["ALIYUN_MODEL_SERVER_KEY", "ALIYUN_API_KEY", "ALIYUN_KEY"]

    base = canonical.upper()
    return [f"{base}_API_KEY", f"{base}_KEY"]


def _resolve_env_api_key(provider_name: str) -> str:
    """Resolve API key from environment variables.

    Args:
        provider_name: Provider name

    Returns:
        API key from environment or empty string if not found
    """
    for env_key in _provider_env_key_candidates(provider_name):
        value = (os.getenv(env_key) or "").strip()
        if value:
            return value
    return ""


class ProviderRegistry:
    """Registry for managing provider adapters and their initialization.

    This class handles:
    - Adapter class lookup
    - API key resolution (encrypted DB keys, environment variables)
    - Base URL resolution
    - Adapter initialization
    - Model discovery
    """

    def __init__(
        self,
        adapter_registry: Dict[str, Type[ProviderBase]],
        encryption_service: EncryptionService,
        provider_config_resolver: Optional[Any] = None,
    ):
        """Initialize provider registry.

        Args:
            adapter_registry: Map of provider names to adapter classes
            encryption_service: Service for decrypting API keys
            provider_config_resolver: Optional callable to resolve provider config
                                     (e.g., base URLs). If not provided, uses default.
        """
        self.adapter_registry = adapter_registry
        self.encryption_service = encryption_service
        self.provider_config_resolver = provider_config_resolver or self._default_config_resolver

    def _default_config_resolver(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Default configuration resolver using provider_registry.

        Args:
            provider_name: Provider name

        Returns:
            Provider configuration dict or None
        """
        try:
            from ..providers.provider_registry import get_provider_registry

            registry = get_provider_registry()
            cfg = registry.get_provider_config_dict((provider_name or "").lower())
            return cfg
        except Exception as e:
            logger.debug(f"Could not resolve config for {provider_name}: {e}")
            return None

    def get_adapter_class(self, provider_name: str) -> Optional[Type[ProviderBase]]:
        """Get adapter class for a provider.

        Args:
            provider_name: Provider name (case-insensitive)

        Returns:
            Adapter class or None if not found
        """
        return self.adapter_registry.get(provider_name.lower())

    def resolve_api_key(
        self,
        provider_name: str,
        encrypted_key: Optional[str] = None,
        plain_key: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve API key from multiple sources.

        Priority order:
        1. Decrypt encrypted_key if provided
        2. Use plain_key if provided
        3. Check environment variables
        4. Return None for providers that don't require keys (e.g., local Ollama)

        Args:
            provider_name: Provider name
            encrypted_key: Encrypted API key from database
            plain_key: Plain text API key from database

        Returns:
            Resolved API key or None
        """
        # Try encrypted key first
        if encrypted_key:
            try:
                return self.encryption_service.decrypt(encrypted_key)
            except Exception as e:
                logger.warning(f"Failed to decrypt API key for provider {provider_name}: {e}")

        # Fall back to plain API key
        if plain_key:
            return plain_key

        # Check environment variables
        env_key = _resolve_env_api_key(provider_name)
        if env_key:
            return env_key

        # Some providers don't require API keys (local deployments)
        if provider_name.lower() in {"ollama", "ollama_gcp", "llamacpp_gcp"}:
            return None

        logger.warning(f"No API key available for provider {provider_name}")
        return None

    def resolve_base_url(
        self, provider_name: str, db_base_url: Optional[str] = None
    ) -> Optional[str]:
        """Resolve base URL for a provider.

        Args:
            provider_name: Provider name
            db_base_url: Base URL from database

        Returns:
            Resolved base URL or None
        """
        # Use database value if available
        if db_base_url:
            return db_base_url.strip() if isinstance(db_base_url, str) else None

        # Fall back to provider config
        cfg = self.provider_config_resolver(provider_name)
        if cfg:
            base_url = cfg.get("base_url")
            if isinstance(base_url, str) and base_url.strip():
                return base_url.strip()

        return None

    async def initialize_adapter(
        self,
        provider_name: str,
        encrypted_key: Optional[str] = None,
        plain_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Optional[ProviderBase]:
        """Initialize a provider adapter.

        Args:
            provider_name: Provider name
            encrypted_key: Encrypted API key from database
            plain_key: Plain text API key from database
            base_url: Base URL from database

        Returns:
            Initialized adapter instance or None if initialization fails
        """
        # Get adapter class
        adapter_class = self.get_adapter_class(provider_name)
        if not adapter_class:
            logger.warning(f"No adapter found for provider {provider_name}")
            return None

        # Resolve API key
        api_key = self.resolve_api_key(provider_name, encrypted_key, plain_key)

        # Resolve base URL
        resolved_base_url = self.resolve_base_url(provider_name, base_url)

        # Check if API key is required but missing
        if api_key is None and provider_name.lower() not in {
            "ollama",
            "ollama_gcp",
            "llamacpp_gcp",
        }:
            logger.warning(f"Cannot initialize {provider_name}: no API key available")
            return None

        # Initialize adapter
        try:
            adapter = adapter_class(api_key, resolved_base_url)
            return adapter
        except Exception as e:
            logger.error(f"Failed to initialize adapter for {provider_name}: {e}")
            return None

    async def get_provider_models(self, adapter: ProviderBase) -> List[str]:
        """Get available models from a provider adapter.

        Args:
            adapter: Initialized adapter instance

        Returns:
            List of model names
        """
        try:
            models = await adapter.list_models()
            return models if models else []
        except Exception as e:
            logger.warning(f"Failed to list models from adapter: {e}")
            return []
