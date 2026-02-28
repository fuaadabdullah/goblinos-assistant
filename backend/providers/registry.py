"""
Provider registry for unified runtime routing.

This registry is the source used by routing_subsystem. Provider definitions come
from apps/goblin-assistant-root/config/providers.toml to keep runtime config aligned.
"""

from __future__ import annotations

import logging
import os
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

from .base import HealthStatus, ProviderBase

logger = logging.getLogger(__name__)


def _clean_env(value: Optional[str]) -> str:
    """Normalize env strings so accidental newline suffixes do not break URLs."""
    return (value or "").strip()


class ProviderRegistry:
    """Registry for ProviderBase implementations used by routing."""

    def __init__(self, config_file: Optional[str] = None):
        self.providers: Dict[str, ProviderBase] = {}
        self.provider_configs: Dict[str, Dict[str, Any]] = {}
        self.aliases: Dict[str, str] = {}
        self.config_file = (
            config_file
            or os.getenv("PROVIDERS_TOML_PATH")
            or str(Path(__file__).resolve().parents[2] / "config" / "providers.toml")
        )
        self.health_cache_ttl_s = max(1, int(float(os.getenv("PROVIDER_HEALTH_CACHE_TTL_S", "15"))))
        self._health_cache: Dict[str, tuple[float, HealthStatus]] = {}
        self._load_providers()

    def _load_providers(self) -> None:
        """Load provider entries from providers.toml and initialize supported ones."""
        config = self._load_toml_config()
        providers_section = config.get("providers", {})
        if not isinstance(providers_section, dict):
            logger.warning("providers.toml missing [providers.*] sections")
            providers_section = {}

        for provider_id, provider_cfg in providers_section.items():
            if not isinstance(provider_cfg, dict):
                continue

            provider_id = str(provider_id)
            self.provider_configs[provider_id] = provider_cfg
            if not self._is_enabled(provider_id, provider_cfg):
                continue

            import_target = self._resolve_import_target(provider_id)
            if not import_target:
                # Unsupported in unified ProviderBase runtime; keep config for catalog.
                logger.debug("Skipping unsupported provider %s", provider_id)
                continue

            module_name, class_name = import_target
            provider = self._initialize_provider(
                provider_id=provider_id,
                config=provider_cfg,
                module_name=module_name,
                class_name=class_name,
            )
            if provider is not None:
                self.providers[provider_id] = provider
                logger.info("Initialized provider %s", provider_id)

        self._initialize_aliases()

    def _load_toml_config(self) -> Dict[str, Any]:
        if not tomllib:
            logger.warning("tomllib unavailable, provider config cannot be loaded")
            return {}
        path = Path(self.config_file)
        if not path.exists():
            logger.warning("Provider config not found at %s", path)
            return {}
        try:
            with path.open("rb") as f:
                return tomllib.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load providers.toml: %s", exc)
            return {}

    def _is_enabled(self, provider_id: str, cfg: Dict[str, Any]) -> bool:
        env_key = f"PROVIDER_{provider_id.upper().replace('-', '_')}_ENABLED"
        env_override = _clean_env(os.getenv(env_key))
        if env_override:
            return env_override.lower() in {"1", "true", "yes", "on"}

        api_key_env = str(cfg.get("api_key_env", "") or "")
        endpoint = self._resolve_endpoint(provider_id, cfg)
        canonical = provider_id.replace("-", "_")

        if provider_id.startswith(("ollama", "llamacpp")):
            return bool(endpoint)

        if canonical == "azure_openai":
            azure_endpoint = _clean_env(os.getenv("AZURE_OPENAI_ENDPOINT"))
            azure_key = self._resolve_api_key(provider_id, api_key_env)
            if azure_endpoint and ".services.ai.azure.com" in azure_endpoint.lower():
                return bool(azure_endpoint and azure_key)
            deployment = _clean_env(os.getenv("AZURE_DEPLOYMENT_ID"))
            return bool(azure_endpoint and azure_key and deployment)

        if canonical == "aliyun":
            aliyun_endpoint = _clean_env(os.getenv("ALIYUN_MODEL_SERVER_URL")) or endpoint
            if api_key_env:
                return bool(aliyun_endpoint and self._resolve_api_key(provider_id, api_key_env))
            return bool(aliyun_endpoint)

        if api_key_env:
            return bool(self._resolve_api_key(provider_id, api_key_env))
        return bool(endpoint)

    def _resolve_api_key(self, provider_id: str, api_key_env: str) -> str:
        canonical = provider_id.replace("-", "_")
        candidates: list[str] = []
        if api_key_env:
            candidates.append(api_key_env)

        # Backward-compatible env aliases used in deployed environments.
        if canonical.startswith("openrouter"):
            candidates.extend(["OPENROUTER_API_KEY", "OPENROUTER_KEY"])
        if canonical in {"openai", "openai_fallback"}:
            candidates.extend(["OPENAI_API_KEY", "OPENAI_KEY"])
        if canonical in {"groq", "grok"}:
            candidates.extend(["GROQ_API_KEY", "GROK_API_KEY"])

        seen: set[str] = set()
        for env_name in candidates:
            if not env_name or env_name in seen:
                continue
            seen.add(env_name)
            value = _clean_env(os.getenv(env_name))
            if value:
                return value
        return ""

    def _resolve_import_target(self, provider_id: str) -> Optional[tuple[str, str]]:
        canonical = provider_id.replace("-", "_")
        if provider_id.startswith("ollama"):
            return ("ollama", "OllamaProvider")
        if provider_id.startswith("llamacpp"):
            return ("llamacpp", "LlamaCppProvider")
        if canonical in {"azure_openai", "openai_fallback"}:
            return ("openai", "OpenAIProvider")
        if canonical in {"aliyun", "alibaba", "alibaba_cloud"}:
            return ("openai", "OpenAIProvider")
        if canonical.startswith("openrouter"):
            # OpenRouter is OpenAI-compatible; reuse the OpenAI provider implementation.
            return ("openai", "OpenAIProvider")
        if canonical.startswith("together"):
            # Together is OpenAI-compatible.
            return ("openai", "OpenAIProvider")
        if canonical.startswith("deepinfra"):
            # DeepInfra is OpenAI-compatible.
            return ("openai", "OpenAIProvider")
        if provider_id == "openai":
            return ("openai", "OpenAIProvider")
        if provider_id == "siliconeflow":
            return ("openai", "OpenAIProvider")
        if provider_id == "anthropic":
            return ("anthropic", "AnthropicProvider")
        # OpenAI-compatible providers — reuse OpenAIProvider with their base URLs
        if provider_id == "deepseek":
            return ("openai", "OpenAIProvider")
        if provider_id == "grok":
            return ("openai", "OpenAIProvider")
        if provider_id == "moonshot":
            return ("openai", "OpenAIProvider")
        if provider_id in ("runpod", "vastai"):
            # RunPod and VastAI expose OpenAI-compatible endpoints
            return ("openai", "OpenAIProvider")
        if provider_id == "gemini":
            # Gemini supports an OpenAI-compatible REST endpoint
            return ("openai", "OpenAIProvider")
        return None

    def _setup_ollama_kwargs(
        self, kwargs: Dict[str, Any], config: Dict[str, Any]
    ) -> None:
        """Setup kwargs for Ollama providers."""
        api_key_env = str(config.get("api_key_env", "") or "")
        kwargs["api_key"] = _clean_env(os.getenv(api_key_env)) or _clean_env(
            os.getenv("LOCAL_LLM_API_KEY")
        )

    def _setup_llamacpp_kwargs(
        self, kwargs: Dict[str, Any], config: Dict[str, Any]
    ) -> None:
        """Setup kwargs for LlamaCpp providers."""
        api_key_env = str(config.get("api_key_env", "") or "")
        api_key = _clean_env(os.getenv(api_key_env))
        if not api_key:
            api_key = _clean_env(os.getenv("LOCAL_LLM_API_KEY")) or _clean_env(
                os.getenv("GCP_LLM_API_KEY")
            )
        kwargs["api_key"] = api_key

    def _setup_standard_provider_kwargs(
        self, kwargs: Dict[str, Any], provider_id: str, config: Dict[str, Any]
    ) -> bool:
        """Setup kwargs for standard providers. Returns False if API key is missing."""
        api_key_env = str(config.get("api_key_env", "") or "")
        api_key = self._resolve_api_key(provider_id, api_key_env)
        if not api_key:
            return False
        kwargs["api_key"] = api_key
        return True

    def _instantiate_provider(
        self, provider_class, kwargs: Dict[str, Any], provider_id: str
    ) -> Optional[ProviderBase]:
        """Instantiate a provider, handling compatibility with older classes."""
        try:
            return provider_class(**kwargs)
        except TypeError:
            # Older provider classes may not accept every optional kwarg.
            minimal_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in {
                    "api_key",
                    "base_url",
                    "models",
                    "cost_per_token_input",
                    "cost_per_token_output",
                }
            }
            provider = provider_class(**minimal_kwargs)
            # Best-effort provider_id override for older classes.
            if hasattr(provider, "_provider_id"):
                setattr(provider, "_provider_id", provider_id)
            return provider
        except Exception as exc:
            logger.warning("Failed to instantiate provider %s: %s", provider_id, exc)
            return None

    def _initialize_provider(
        self, provider_id: str, config: Dict[str, Any], module_name: str, class_name: str
    ) -> Optional[ProviderBase]:
        try:
            provider_class = self._import_provider_class(module_name, class_name)
        except Exception as exc:
            logger.warning("Provider import failed for %s: %s", provider_id, exc)
            return None

        endpoint = self._resolve_endpoint(provider_id, config)
        models = [str(m) for m in (config.get("models") or []) if m is not None]
        timeout_ms = int(config.get("default_timeout_ms", 12000) or 12000)
        kwargs: Dict[str, Any] = {
            "models": models,
            "provider_id": provider_id,
            "cost_per_token_input": float(config.get("cost_score", 0.0) or 0.0),
            "cost_per_token_output": 0.0,
            "timeout_seconds": max(1, timeout_ms // 1000),
            "base_url": endpoint,
        }

        # Setup provider-specific kwargs
        if provider_id.startswith("ollama"):
            self._setup_ollama_kwargs(kwargs, config)
        elif provider_id.startswith("llamacpp"):
            self._setup_llamacpp_kwargs(kwargs, config)
        else:
            if not self._setup_standard_provider_kwargs(kwargs, provider_id, config):
                return None

        return self._instantiate_provider(provider_class, kwargs, provider_id)

    def _import_provider_class(self, module_name: str, class_name: str):
        candidates = []
        if __package__:
            candidates.append(f"{__package__}.{module_name}")
        candidates.append(f"backend.providers.{module_name}")
        candidates.append(f"providers.{module_name}")

        last_exc: Optional[Exception] = None
        for candidate in candidates:
            try:
                module = import_module(candidate)
                return getattr(module, class_name)
            except Exception as exc:  # pragma: no cover - import path dependent
                last_exc = exc
                continue
        raise RuntimeError(last_exc or f"Could not import {module_name}.{class_name}")

    def _resolve_endpoint(self, provider_id: str, cfg: Dict[str, Any]) -> str:
        endpoint = _clean_env(str(cfg.get("endpoint", "") or ""))

        # Preferred config-driven endpoint env resolution from providers.toml.
        endpoint_env = str(cfg.get("endpoint_env", "") or "").strip()
        if endpoint_env:
            endpoint_from_env = _clean_env(os.getenv(endpoint_env))
            if endpoint_from_env:
                endpoint = endpoint_from_env

        # Backward-compatible fallback env key support.
        if not endpoint:
            endpoint_fallback_env = str(cfg.get("endpoint_fallback", "") or "").strip()
            if endpoint_fallback_env:
                endpoint = _clean_env(os.getenv(endpoint_fallback_env))

        endpoint_overrides = {
            "ollama": "OLLAMA_BASE_URL",
            "ollama-gcp": "OLLAMA_GCP_URL",
            "ollama_gcp": "OLLAMA_GCP_URL",
            "ollama_kamatera": "KAMATERA_SERVER1_URL",
            "llamacpp-gcp": "LLAMACPP_GCP_URL",
            "llamacpp_gcp": "LLAMACPP_GCP_URL",
            "llamacpp_kamatera": "KAMATERA_SERVER1_URL",
            "azure-openai": "AZURE_OPENAI_ENDPOINT",
            "azure_openai": "AZURE_OPENAI_ENDPOINT",
            "aliyun": "ALIYUN_MODEL_SERVER_URL",
            "siliconeflow": "SILICONEFLOW_BASE_URL",
            "deepseek": "DEEPSEEK_BASE_URL",
            "grok": "XAI_BASE_URL",
            "moonshot": "MOONSHOT_BASE_URL",
            "runpod": "RUNPOD_BASE_URL",
            "vastai": "VASTAI_BASE_URL",
            "gemini": "GEMINI_BASE_URL",
        }
        env_name = endpoint_overrides.get(provider_id)
        if env_name:
            override = _clean_env(os.getenv(env_name))
            if override:
                endpoint = override
        return endpoint.rstrip("/")

    def _initialize_aliases(self) -> None:
        """Canonical-only mode: legacy provider aliases removed."""
        self.aliases = {}

    def _resolve_provider_key(self, provider_id: str) -> str:
        """Resolve provider key - returns the provider_id if registered."""
        return provider_id

    def get_provider_health_map(self, refresh: bool = True) -> Dict[str, HealthStatus]:
        """
        Return provider health keyed by registry provider key.

        Keying by registry key (not provider.provider_id) avoids collisions when
        multiple configured entries share the same provider implementation.
        """
        health: Dict[str, HealthStatus] = {}
        now = time.time()
        for provider_key, provider in self.providers.items():
            cached = self._health_cache.get(provider_key)
            if cached and (now - cached[0]) < self.health_cache_ttl_s:
                status = cached[1]
            elif not refresh:
                # Unknown health state until a refresh/check runs.
                continue
            else:
                try:
                    status = provider.health_check()
                except Exception as exc:
                    logger.warning("Health check failed for %s: %s", provider_key, exc)
                    status = HealthStatus.UNHEALTHY
                self._health_cache[provider_key] = (now, status)
            health[provider_key] = status
        return health

    def get_available_providers(self) -> List[ProviderBase]:
        """Get healthy providers with short-lived health cache."""
        available: List[ProviderBase] = []
        health_map = self.get_provider_health_map(refresh=True)
        for provider_key, provider in self.providers.items():
            if health_map.get(provider_key) == HealthStatus.HEALTHY:
                available.append(provider)
        return available

    def get_provider(self, provider_id: str) -> Optional[ProviderBase]:
        key = self._resolve_provider_key(provider_id)
        provider = self.providers.get(key)
        if provider:
            return provider
        # Fallback: resolve by provider object's provider_id value.
        for item in self.providers.values():
            if getattr(item, "provider_id", None) == provider_id:
                return item
        return None

    def get_providers_by_capability(self, capability: str) -> List[ProviderBase]:
        providers: List[ProviderBase] = []
        for provider in self.get_available_providers():
            caps = provider.capabilities
            if caps.get(f"supports_{capability}", False):
                providers.append(provider)
                continue
            if capability in caps.get("capabilities", []):
                providers.append(provider)
        return providers

    def get_providers_for_model(self, model: str) -> List[ProviderBase]:
        providers: List[ProviderBase] = []
        for provider in self.get_available_providers():
            if model in provider.capabilities.get("models", []):
                providers.append(provider)
        return providers

    def get_provider_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Return model and endpoint metadata for exposed provider routes."""
        catalog: Dict[str, Dict[str, Any]] = {}
        for provider_id, cfg in self.provider_configs.items():
            models = [str(m) for m in (cfg.get("models") or []) if m is not None]
            catalog[provider_id] = {
                "provider_id": provider_id,
                "name": str(cfg.get("name", provider_id)),
                "endpoint": self._resolve_endpoint(provider_id, cfg),
                "models": models,
                "capabilities": [str(c) for c in (cfg.get("capabilities") or [])],
            }
        return catalog

    def reload_providers(self) -> None:
        self.providers.clear()
        self.provider_configs.clear()
        self.aliases.clear()
        self._health_cache.clear()
        self._load_providers()
        logger.info("Reloaded providers from %s", self.config_file)


_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
