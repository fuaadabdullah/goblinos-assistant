"""
Provider Registry for centralized configuration management.

Centralizes all provider configuration including hosts, costs, latency thresholds,
and other provider-specific settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from urllib.parse import urlparse

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider operational status."""

    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""

    name: str
    host: str
    api_key_env: str
    base_url: Optional[str] = None
    timeout_seconds: int = 30
    max_retries: int = 2
    cost_per_token_input: float = 0.0
    cost_per_token_output: float = 0.0
    latency_threshold_ms: int = 5000  # 5 seconds
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    status: ProviderStatus = ProviderStatus.ACTIVE
    models: List[str] = None
    capabilities: List[str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = []
        if self.capabilities is None:
            self.capabilities = ["chat"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        """Create from dictionary."""
        data["status"] = ProviderStatus(data["status"])
        return cls(**data)


class ProviderRegistry:
    """Central registry for all provider configurations."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the provider registry.

        Args:
            config_file: Path to JSON config file (optional)
        """
        self.providers: Dict[str, ProviderConfig] = {}
        self.aliases: Dict[str, str] = {}
        self.config_file = config_file or os.getenv("PROVIDERS_TOML_PATH") or str(
            Path(__file__).resolve().parents[2] / "config" / "providers.toml"
        )
        self._load_config()
        self._initialize_aliases()

    def _load_config(self):
        """Load configuration from file or use defaults."""
        path = Path(self.config_file)
        if path.exists():
            try:
                if path.suffix == ".toml":
                    self._load_from_toml(path)
                else:
                    # Backward compatible path; prefer TOML at runtime.
                    self._load_defaults()
                logger.info(f"Loaded provider config from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}, using defaults")
                self._load_defaults()
        else:
            logger.info("No config file found, using defaults")
            self._load_defaults()

    def _load_from_toml(self, path: Path):
        """Load provider config from providers.toml."""
        if not tomllib:
            raise RuntimeError("tomllib is required to parse providers.toml")
        with path.open("rb") as f:
            data = tomllib.load(f)
        provider_sections = data.get("providers", {})
        if not isinstance(provider_sections, dict):
            raise RuntimeError("providers.toml missing [providers.*] sections")

        self.providers.clear()
        for provider_id, cfg in provider_sections.items():
            if not isinstance(cfg, dict):
                continue
            endpoint = str(cfg.get("endpoint", "") or "").strip()
            parsed = urlparse(endpoint)
            host = parsed.netloc or endpoint
            status_raw = str(cfg.get("status", "active") or "active").lower()
            status = ProviderStatus.ACTIVE
            if status_raw in {"degraded", "maintenance", "disabled"}:
                status = ProviderStatus(status_raw)

            self.providers[str(provider_id)] = ProviderConfig(
                name=str(provider_id),
                host=host,
                api_key_env=str(cfg.get("api_key_env", "") or ""),
                base_url=endpoint or None,
                timeout_seconds=max(1, int((cfg.get("default_timeout_ms", 30000) or 30000) / 1000)),
                max_retries=2,
                cost_per_token_input=float(cfg.get("cost_score", 0.0) or 0.0),
                cost_per_token_output=0.0,
                latency_threshold_ms=int(cfg.get("default_timeout_ms", 30000) or 30000),
                models=[str(m) for m in (cfg.get("models") or []) if m is not None],
                capabilities=[str(c) for c in (cfg.get("capabilities") or ["chat"]) if c is not None],
                status=status,
            )

    def _load_defaults(self):
        """Load default provider configurations."""
        defaults = {
            "openai": ProviderConfig(
                name="openai",
                host="api.openai.com",
                api_key_env="OPENAI_API_KEY",
                base_url="https://api.openai.com/v1",
                cost_per_token_input=0.0015,
                cost_per_token_output=0.002,
                latency_threshold_ms=3000,
                models=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                capabilities=["chat", "embeddings", "vision"],
            ),
            "anthropic": ProviderConfig(
                name="anthropic",
                host="api.anthropic.com",
                api_key_env="ANTHROPIC_API_KEY",
                base_url="https://api.anthropic.com",
                cost_per_token_input=0.008,
                cost_per_token_output=0.024,
                latency_threshold_ms=4000,
                models=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                capabilities=["chat", "vision"],
            ),
            "ollama": ProviderConfig(
                name="ollama",
                host="localhost:11434",
                api_key_env="OLLAMA_API_KEY",
                base_url="http://localhost:11434",
                cost_per_token_input=0.0,
                cost_per_token_output=0.0,
                latency_threshold_ms=10000,
                models=["llama2", "codellama", "mistral"],
                capabilities=["chat", "embeddings"],
            ),
            "groq": ProviderConfig(
                name="groq",
                host="api.groq.com",
                api_key_env="GROQ_API_KEY",
                base_url="https://api.groq.com/openai/v1",
                cost_per_token_input=0.0001,
                cost_per_token_output=0.0001,
                latency_threshold_ms=2000,
                models=["llama2-70b", "mixtral-8x7b"],
                capabilities=["chat"],
            ),
            "deepseek": ProviderConfig(
                name="deepseek",
                host="api.deepseek.com",
                api_key_env="DEEPSEEK_API_KEY",
                base_url="https://api.deepseek.com",
                cost_per_token_input=0.0001,
                cost_per_token_output=0.0002,
                latency_threshold_ms=3000,
                models=["deepseek-chat", "deepseek-coder"],
                capabilities=["chat"],
            ),
            "vertex": ProviderConfig(
                name="vertex",
                host="us-central1-aiplatform.googleapis.com",
                api_key_env="GOOGLE_CLOUD_API_KEY",
                base_url="https://us-central1-aiplatform.googleapis.com",
                cost_per_token_input=0.00125,
                cost_per_token_output=0.005,
                latency_threshold_ms=3000,
                models=[
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "gemini-pro",
                    "gemini-pro-vision",
                ],
                capabilities=["chat", "vision"],
            ),
        }

        self.providers.update(defaults)

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get provider configuration by name.

        Args:
            name: Provider name

        Returns:
            ProviderConfig or None if not found
        """
        key = self.aliases.get(name, name)
        if key in self.providers:
            return self.providers[key]
        normalized = name.replace("-", "_")
        key = self.aliases.get(normalized, normalized)
        return self.providers.get(key)

    def get_all_providers(self) -> Dict[str, ProviderConfig]:
        """Get all provider configurations.

        Returns:
            Dict of provider name -> ProviderConfig
        """
        return self.providers.copy()

    def get_active_providers(self) -> Dict[str, ProviderConfig]:
        """Get only active providers.

        Returns:
            Dict of active provider name -> ProviderConfig
        """
        return {
            name: config
            for name, config in self.providers.items()
            if config.status == ProviderStatus.ACTIVE
        }

    def update_provider_status(self, name: str, status: ProviderStatus):
        """Update provider operational status.

        Args:
            name: Provider name
            status: New status
        """
        if name in self.providers:
            self.providers[name].status = status
            logger.info(f"Updated {name} status to {status.value}")
        else:
            logger.warning(f"Provider {name} not found for status update")

    def get_provider_config_dict(self, name: str) -> Optional[Dict[str, Any]]:
        """Get provider config as dictionary for adapter initialization.

        Args:
            name: Provider name

        Returns:
            Config dict or None if provider not found
        """
        provider = self.get_provider(name)
        if not provider:
            return None

        api_key = (os.getenv(provider.api_key_env) or "").strip() if provider.api_key_env else ""
        if not api_key:
            logger.warning(f"No API key found for {name} ({provider.api_key_env})")
            # Don't return None - some providers (local) don't need API keys
            api_key = ""

        # Check for environment variable overrides for base_url
        # This allows dynamic configuration of GCP, Kamatera, and other cloud instances
        base_url = provider.base_url
        env_url_mappings = {
            "ollama_gcp": "OLLAMA_GCP_URL",
            "kamatera": "KAMATERA_SERVER1_URL",
            "llamacpp_gcp": "LLAMACPP_GCP_URL",
            "runpod": "RUNPOD_BASE_URL",
            "vastai": "VASTAI_BASE_URL",
            "ollama": "OLLAMA_BASE_URL",
            "siliconeflow": "SILICONEFLOW_BASE_URL",
            "aliyun": "ALIYUN_MODEL_SERVER_URL",
            "azure_openai": "AZURE_OPENAI_ENDPOINT",
        }

        if name in env_url_mappings:
            env_url = os.getenv(env_url_mappings[name])
            if env_url:
                base_url = env_url.strip()
                logger.info(
                    f"Using environment override for {name} base_url: {base_url}"
                )

        if isinstance(base_url, str):
            base_url = base_url.strip()

        return {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": provider.timeout_seconds,
            "retries": provider.max_retries,
            "cost_per_token_input": provider.cost_per_token_input,
            "cost_per_token_output": provider.cost_per_token_output,
            "latency_threshold_ms": provider.latency_threshold_ms,
        }

    def save_config(self):
        """No-op for TOML runtime source to avoid drift from providers.toml."""
        logger.info("Provider config is sourced from providers.toml; save is skipped.")

    def add_provider(self, config: ProviderConfig):
        """Add a new provider configuration.

        Args:
            config: ProviderConfig to add
        """
        self.providers[config.name] = config
        logger.info(f"Added provider {config.name}")

    def remove_provider(self, name: str):
        """Remove a provider configuration.

        Args:
            name: Provider name to remove
        """
        if name in self.providers:
            del self.providers[name]
            logger.info(f"Removed provider {name}")
        else:
            logger.warning(f"Provider {name} not found for removal")

    def _initialize_aliases(self):
        """Canonical-only mode: legacy provider aliases removed."""
        self.aliases = {}


# Global registry instance
_registry_instance: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ProviderRegistry()
    return _registry_instance
