"""
LLM Environment Configuration Loader

Bridges environment-based LLM configuration (.env) with Goblin Assistant routing.
Dynamically loads API keys and configures providers from environment variables.

This module integrates Alibaba Cloud (Qwen-Coder), Azure OpenAI, and alternative
LLM providers into the Goblin Assistant provider routing system.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    _env_file = Path(".env")
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass


@dataclass
class LLMProviderConfig:
    """LLM Provider configuration loaded from environment"""

    name: str
    endpoint: str
    model: str
    api_key: Optional[str] = None
    priority: int = 1
    is_configured: bool = False
    capabilities: List[str] = field(default_factory=lambda: ["chat", "completions"])

    def __post_init__(self):
        """Validate after init"""
        self.is_configured = bool(self.api_key) if self.api_key is not None else True


class LLMEnvLoader:
    """Load and validate LLM provider configuration from environment"""

    # Provider configuration mapping
    PROVIDER_CONFIGS = {
        "together": {
            "endpoint": "https://api.together.xyz/v1",
            "env_var": "TOGETHER_API_KEY",
            "default_model": "qwen2.5-coder-32b-instruct",
            "priority": 1,
            "capabilities": ["chat", "completions", "code-generation"],
        },
        "azure": {
            "endpoint": "https://{resource}.openai.azure.com/",
            "env_var": "AZURE_API_KEY",
            "default_model": "gpt-4",
            "priority": 1,
            "requires_env": ["AZURE_OPENAI_ENDPOINT"],
            "capabilities": ["chat", "completions", "code-generation"],
        },
        "openrouter": {
            "endpoint": "https://openrouter.ai/api/v1",
            "env_var": "OPENROUTER_API_KEY",
            "default_model": "qwen/qwen-2.5-coder-32b-instruct",
            "priority": 2,
            "capabilities": ["chat", "completions", "code-generation"],
        },
        "deepinfra": {
            "endpoint": "https://api.deepinfra.com/v1",
            "env_var": "DEEPINFRA_API_KEY",
            "default_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "priority": 2,
            "capabilities": ["chat", "completions", "code-generation"],
        },
        "groq": {
            "endpoint": "https://api.groq.com/openai/v1",
            "env_var": "GROQ_API_KEY",
            "default_model": "qwen-2.5-coder-32b-instruct",
            "priority": 2,
            "capabilities": ["chat", "completions", "code-generation"],
        },
        "aliyun": {
            "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "endpoint_env": "ALIYUN_MODEL_SERVER_URL",
            "env_vars": ["ALIYUN_MODEL_SERVER_KEY", "ALIYUN_API_KEY", "ALIYUN_KEY"],
            "default_model": "qwen-plus",
            "priority": 1,
            "capabilities": ["chat", "completions", "code-generation"],
        },
        "openai": {
            "endpoint": "https://api.openai.com/v1",
            "env_var": "OPENAI_API_KEY",
            "default_model": "gpt-4",
            "priority": 3,
            "capabilities": ["chat", "completions"],
        },
        "ollama_gcp": {
            "endpoint_env": "OLLAMA_GCP_URL",
            "endpoint_env_fallback": "GCP_OLLAMA_URL",
            "env_var": None,  # No API key needed for Ollama
            "default_model": "qwen2.5:3b",
            "priority": 1,
            "capabilities": ["chat", "completions", "code-generation"],
        },
        "llamacpp_gcp": {
            "endpoint_env": "LLAMACPP_GCP_URL",
            "endpoint_env_fallback": "GCP_LLAMACPP_URL",
            "env_var": None,  # No API key needed for llama.cpp
            "default_model": "phi-3-mini-4k-instruct-q4",
            "priority": 1,
            "capabilities": ["chat", "completions", "code-generation"],
        },
    }

    def __init__(self):
        """Initialize the LLM loader"""
        self.configured_providers: Dict[str, LLMProviderConfig] = {}
        self._load_configured_providers()

    def _load_configured_providers(self) -> None:
        """Load all configured providers from environment"""
        for provider_name, config in self.PROVIDER_CONFIGS.items():
            api_key = None

            # Get API key if required
            if config.get("env_vars"):
                for env_name in config.get("env_vars", []):
                    value = os.getenv(env_name or "")
                    if value:
                        api_key = value
                        break
            elif config.get("env_var"):
                api_key = os.getenv(config["env_var"])

            # Get endpoint (may need environment variable for Azure or GCP)
            endpoint = config.get("endpoint", "")

            if provider_name == "azure":
                endpoint = os.getenv(
                    "AZURE_OPENAI_ENDPOINT", "https://{resource}.openai.azure.com/"
                )
            elif config.get("endpoint_env"):
                # Handle GCP providers with environment variable endpoints
                endpoint = (
                    os.getenv(config["endpoint_env"])
                    or os.getenv(config.get("endpoint_env_fallback", ""))
                    or endpoint
                )

            # Create provider config
            provider = LLMProviderConfig(
                name=provider_name,
                endpoint=endpoint,
                api_key=api_key,
                model=os.getenv(f"{provider_name.upper()}_MODEL", config["default_model"]),
                priority=config.get("priority", 2),
                capabilities=config.get("capabilities", ["chat", "completions"]),
            )

            self.configured_providers[provider_name] = provider

            if provider.is_configured:
                logger.info(f"✅ LLM Provider configured: {provider_name}")
            else:
                logger.debug(f"⏭️  LLM Provider not configured: {provider_name}")

    def get_available_providers(self) -> Dict[str, LLMProviderConfig]:
        """Get all configured (API key set) providers"""
        return {
            name: provider
            for name, provider in self.configured_providers.items()
            if provider.is_configured
        }

    def get_all_providers(self) -> Dict[str, LLMProviderConfig]:
        """Get all providers (configured or not)"""
        return self.configured_providers.copy()

    def get_provider(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Get a specific provider by name"""
        return self.configured_providers.get(provider_name)

    def get_provider_for_capability(
        self, capability: str, prefer_cost: bool = False
    ) -> Optional[LLMProviderConfig]:
        """
        Get best provider for a capability.

        Args:
            capability: Required capability (e.g., 'code-generation')
            prefer_cost: If True, prefer cheaper providers

        Returns:
            Best provider configuration or None
        """
        candidates = [
            provider
            for provider in self.get_available_providers().values()
            if capability in provider.capabilities
        ]

        if not candidates:
            return None

        # Sort by priority (lower number = better)
        candidates.sort(key=lambda p: p.priority)

        return candidates[0]

    def get_provider_headers(self, provider_name: str) -> Dict[str, str]:
        """
        Get HTTP headers for provider API calls.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary of headers
        """
        provider = self.configured_providers.get(provider_name)
        if not provider or not provider.api_key:
            return {}

        if provider_name == "azure":
            return {"api-key": provider.api_key}
        else:
            return {"Authorization": f"Bearer {provider.api_key}"}

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate LLM configuration.

        Returns:
            Validation report
        """
        available = self.get_available_providers()
        configured_count = len(available)

        report = {
            "total_providers": len(self.configured_providers),
            "configured_providers": configured_count,
            "providers": {},
            "status": "configured" if configured_count > 0 else "not_configured",
        }

        for name, provider in self.configured_providers.items():
            report["providers"][name] = {
                "configured": provider.is_configured,
                "endpoint": provider.endpoint,
                "model": provider.model,
                "priority": provider.priority,
                "capabilities": provider.capabilities,
            }

        return report

    def get_fallback_chain(self, capability: str = "chat") -> List[str]:
        """
        Get fallback chain of providers for a capability.
        Ordered by priority.

        Args:
            capability: Required capability

        Returns:
            List of provider names in fallback order
        """
        candidates = [
            (provider.name, provider.priority)
            for provider in self.get_available_providers().values()
            if capability in provider.capabilities
        ]

        candidates.sort(key=lambda x: x[1])
        return [name for name, _ in candidates]


# Singleton instance
_loader = None


def get_llm_loader() -> LLMEnvLoader:
    """Get or create LLM loader instance"""
    global _loader
    if _loader is None:
        _loader = LLMEnvLoader()

        # Log configuration
        report = _loader.validate_configuration()
        if report["configured_providers"] > 0:
            logger.info(
                f"✅ LLM Integration: {report['configured_providers']} "
                f"provider(s) configured and ready"
            )
        else:
            logger.warning("⚠️  No LLM providers configured - please set API keys in .env file")

    return _loader


def validate_llm_setup() -> bool:
    """Quick validation of LLM setup"""
    loader = get_llm_loader()
    report = loader.validate_configuration()
    return report["configured_providers"] > 0
