"""
Configuration for the routing service.

This module centralizes routing configuration that was previously hardcoded
in the RoutingService class, making it easier to test and override.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Type, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RoutingConfig:
    """Configuration for intelligent routing service.

    Attributes:
        sla_targets: SLA target response times in milliseconds for each priority level
        cost_budget_weights: Weights for scoring factors (latency, cost, SLA compliance)
        adapter_registry: Mapping of provider names to adapter classes
        encryption_key: Key for encrypting/decrypting API keys (loaded from env)
    """

    # SLA targets in milliseconds for each priority level
    sla_targets: Dict[str, float] = field(
        default_factory=lambda: {
            "ultra_low": 500,  # ms
            "low": 1000,  # ms
            "medium": 2000,  # ms
            "high": 5000,  # ms
        }
    )

    # Weights for multi-factor scoring
    cost_budget_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "latency_priority": 0.3,  # Weight for latency in scoring
            "cost_priority": 0.4,  # Weight for cost in scoring
            "sla_compliance": 0.3,  # Weight for SLA compliance
        }
    )

    # Provider adapter registry (will be populated with actual classes)
    adapter_registry: Dict[str, Type] = field(default_factory=dict)

    # Encryption key for API keys (loaded from environment)
    encryption_key: str = field(default="")

    @classmethod
    def from_env(cls, encryption_key: Optional[str] = None) -> "RoutingConfig":
        """Create configuration from environment variables.

        Args:
            encryption_key: Optional encryption key override. If not provided,
                          loads from ROUTING_ENCRYPTION_KEY environment variable.

        Returns:
            RoutingConfig instance with values loaded from environment

        Raises:
            ValueError: If required environment variables are missing
        """
        # Import adapters here to avoid circular imports
        from ..providers import (
            OpenAIAdapter,
            AnthropicAdapter,
            GrokAdapter,
            DeepSeekAdapter,
            OllamaAdapter,
            LlamaCppAdapter,
            TinyLlamaAdapter,
            SiliconeflowAdapter,
            MoonshotAdapter,
            ElevenLabsAdapter,
        )

        # Get encryption key from parameter or environment
        resolved_key = encryption_key or os.getenv("ROUTING_ENCRYPTION_KEY", "")
        if not resolved_key:
            logger.warning(
                "ROUTING_ENCRYPTION_KEY not set. API key decryption will fail."
            )

        # Build adapter registry
        adapter_registry = {
            "openai": OpenAIAdapter,
            "openai-fallback": OpenAIAdapter,
            "anthropic": AnthropicAdapter,
            "grok": GrokAdapter,
            "deepseek": DeepSeekAdapter,
            "openrouter": OpenAIAdapter,  # OpenRouter uses OpenAI format
            "azure-openai": OpenAIAdapter,
            "azure_openai": OpenAIAdapter,
            "aliyun": OpenAIAdapter,
            "ollama_gcp": OllamaAdapter,
            "llamacpp_gcp": LlamaCppAdapter,
            "tinylama": TinyLlamaAdapter,
            "siliconeflow": SiliconeflowAdapter,
            "moonshot": MoonshotAdapter,
            "elevenlabs": ElevenLabsAdapter,
        }

        # Allow environment-based override of SLA targets
        sla_targets = {
            "ultra_low": float(os.getenv("SLA_ULTRA_LOW_MS", "500")),
            "low": float(os.getenv("SLA_LOW_MS", "1000")),
            "medium": float(os.getenv("SLA_MEDIUM_MS", "2000")),
            "high": float(os.getenv("SLA_HIGH_MS", "5000")),
        }

        # Allow environment-based override of scoring weights
        cost_budget_weights = {
            "latency_priority": float(os.getenv("WEIGHT_LATENCY", "0.3")),
            "cost_priority": float(os.getenv("WEIGHT_COST", "0.4")),
            "sla_compliance": float(os.getenv("WEIGHT_SLA", "0.3")),
        }

        return cls(
            sla_targets=sla_targets,
            cost_budget_weights=cost_budget_weights,
            adapter_registry=adapter_registry,
            encryption_key=resolved_key,
        )

    def get_sla_target(self, priority: str) -> float:
        """Get SLA target for a given priority level.

        Args:
            priority: Priority level (ultra_low, low, medium, high)

        Returns:
            SLA target in milliseconds
        """
        return self.sla_targets.get(priority, self.sla_targets["medium"])

    def get_adapter_class(self, provider_name: str) -> Type:
        """Get adapter class for a provider.

        Args:
            provider_name: Provider name (case-insensitive)

        Returns:
            Adapter class or None if not found
        """
        return self.adapter_registry.get(provider_name.lower())  # type: ignore

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Check SLA targets are positive
        for priority, target in self.sla_targets.items():
            if target <= 0:
                raise ValueError(
                    f"SLA target for {priority} must be positive, got {target}"
                )

        # Check weights sum to approximately 1.0
        weight_sum = sum(self.cost_budget_weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            logger.warning(
                f"Cost budget weights sum to {weight_sum:.2f}, expected ~1.0. "
                "Scoring may produce unexpected results."
            )

        # Check adapter registry is not empty
        if not self.adapter_registry:
            raise ValueError("Adapter registry is empty")

        return True


# Default instance for backward compatibility
_default_config: Optional[RoutingConfig] = None


def get_default_config() -> RoutingConfig:
    """Get the default routing configuration.

    This provides a singleton-like default config for backward compatibility
    with code that doesn't explicitly inject configuration.

    Returns:
        Default RoutingConfig instance
    """
    global _default_config
    if _default_config is None:
        _default_config = RoutingConfig.from_env()
        _default_config.validate()
    return _default_config
