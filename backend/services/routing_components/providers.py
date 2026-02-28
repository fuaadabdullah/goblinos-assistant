"""Provider discovery helpers for routing."""

import logging
import os
from typing import Any

try:
    from backend.models.provider import Provider
    from backend.providers.provider_registry import (
        get_provider_registry as get_provider_settings_registry,
    )
except ImportError:  # pragma: no cover - fallback for local module execution
    from models.provider import Provider
    from providers.provider_registry import (
        get_provider_registry as get_provider_settings_registry,
    )

logger = logging.getLogger(__name__)


def _provider_env_key_candidates(provider_name: str) -> list[str]:
    normalized = (provider_name or "").lower().replace("-", "_")
    candidates: list[str] = []
    if normalized in {"aliyun", "alibaba", "alibaba_cloud"}:
        candidates.extend(
            ["ALIYUN_MODEL_SERVER_KEY", "ALIYUN_API_KEY", "ALIYUN_KEY"]
        )
    if normalized in {"azure_openai", "azure"}:
        candidates.extend(["AZURE_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY"])

    base = (provider_name or "").upper().replace("-", "_")
    candidates.extend([f"{base}_API_KEY", f"{base}_KEY"])

    unique: list[str] = []
    seen: set[str] = set()
    for env_key in candidates:
        if env_key in seen:
            continue
        seen.add(env_key)
        unique.append(env_key)
    return unique


def _resolve_env_api_key(provider_name: str) -> str:
    for env_key in _provider_env_key_candidates(provider_name):
        value = (os.getenv(env_key) or "").strip()
        if value:
            return value
    return ""


def _resolve_provider_base_url(provider_name: str) -> str | None:
    try:
        cfg = get_provider_settings_registry().get_provider_config_dict(
            (provider_name or "").lower()
        )
        base_url = (cfg or {}).get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            return base_url.strip()
    except Exception:
        return None
    return None


async def discover_providers(
    db,
    encryption_service,
    adapters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Discover all active providers and their capabilities.

    Returns:
        List of provider information dictionaries
    """
    import asyncio

    def _sync_query():
        return db.query(Provider).filter(Provider.is_active).all()

    providers = await asyncio.to_thread(_sync_query)

    result = []
    for provider in providers:
        api_key = None
        if provider.api_key_encrypted:
            try:
                api_key = encryption_service.decrypt(provider.api_key_encrypted)
            except Exception as e:
                logger.warning(f"Failed to decrypt API key for provider {provider.name}: {e}")
                api_key = None

        if not api_key and provider.api_key:
            api_key = provider.api_key

        if not api_key:
            env_key = _resolve_env_api_key(provider.name)
            api_key = env_key or None

        if not api_key and provider.name.lower() not in {
            "ollama",
            "ollama_gcp",
            "llamacpp_gcp",
        }:
            logger.warning(f"No API key available for provider {provider.name}")
            continue

        adapter_class = adapters.get(provider.name.lower())
        if not adapter_class:
            logger.warning(f"No adapter found for provider {provider.name}")
            continue

        base_url = getattr(provider, "base_url", None) or None
        if not base_url:
            base_url = _resolve_provider_base_url(provider.name)

        adapter = adapter_class(api_key, base_url)
        models = await adapter.list_models()

        result.append(
            {
                "id": provider.id,
                "name": provider.name,
                "display_name": provider.display_name,
                "base_url": base_url,
                "capabilities": provider.capabilities,
                "models": models,
                "priority": provider.priority,
                "is_active": provider.is_active,
            }
        )

    return result
