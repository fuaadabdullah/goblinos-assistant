"""
Tests for routing_config module.

Validates configuration loading, defaults, and environment variable support.
"""

import os
import pytest
from unittest.mock import patch

from backend.services.routing_config import RoutingConfig, get_default_config


class TestRoutingConfig:
    """Test suite for RoutingConfig dataclass."""

    def test_default_sla_targets(self):
        """Test default SLA target values."""
        config = RoutingConfig()

        assert config.sla_targets["ultra_low"] == 500
        assert config.sla_targets["low"] == 1000
        assert config.sla_targets["medium"] == 2000
        assert config.sla_targets["high"] == 5000

    def test_default_cost_weights(self):
        """Test default cost budget weight values."""
        config = RoutingConfig()

        assert config.cost_budget_weights["latency_priority"] == 0.3
        assert config.cost_budget_weights["cost_priority"] == 0.4
        assert config.cost_budget_weights["sla_compliance"] == 0.3

    def test_get_sla_target(self):
        """Test SLA target retrieval."""
        config = RoutingConfig()

        assert config.get_sla_target("ultra_low") == 500
        assert config.get_sla_target("low") == 1000
        assert config.get_sla_target("invalid") == 2000  # Falls back to medium

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = RoutingConfig(
            sla_targets={"ultra_low": 100, "low": 200, "medium": 300, "high": 400},
            cost_budget_weights={
                "latency_priority": 0.33,
                "cost_priority": 0.33,
                "sla_compliance": 0.34,
            },
            adapter_registry={"openai": object},  # Mock adapter
        )

        assert config.validate() is True

    def test_validate_negative_sla_target(self):
        """Test validation fails for negative SLA targets."""
        config = RoutingConfig(
            sla_targets={"ultra_low": -100, "low": 200, "medium": 300, "high": 400},
        )

        with pytest.raises(ValueError, match="SLA target.*must be positive"):
            config.validate()

    def test_validate_empty_adapter_registry(self):
        """Test validation fails for empty adapter registry."""
        config = RoutingConfig(
            adapter_registry={},
        )

        with pytest.raises(ValueError, match="Adapter registry is empty"):
            config.validate()

    @patch.dict(
        os.environ,
        {
            "ROUTING_ENCRYPTION_KEY": "test_key_32_characters_long_123",
            "SLA_ULTRA_LOW_MS": "300",
            "SLA_LOW_MS": "800",
            "WEIGHT_LATENCY": "0.5",
            "WEIGHT_COST": "0.3",
            "WEIGHT_SLA": "0.2",
        },
    )
    def test_from_env_with_overrides(self):
        """Test configuration loading from environment variables."""
        config = RoutingConfig.from_env()

        # Check encryption key loaded
        assert config.encryption_key == "test_key_32_characters_long_123"

        # Check SLA targets loaded from env
        assert config.sla_targets["ultra_low"] == 300
        assert config.sla_targets["low"] == 800

        # Check weights loaded from env
        assert config.cost_budget_weights["latency_priority"] == 0.5
        assert config.cost_budget_weights["cost_priority"] == 0.3
        assert config.cost_budget_weights["sla_compliance"] == 0.2

        # Check adapter registry populated
        assert "openai" in config.adapter_registry
        assert "anthropic" in config.adapter_registry
        assert "deepseek" in config.adapter_registry

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_without_encryption_key(self):
        """Test configuration loading works when encryption key missing."""
        config = RoutingConfig.from_env()

        # Should still create config but with empty encryption key
        assert config.encryption_key == ""

    def test_get_adapter_class(self):
        """Test adapter class retrieval."""
        from backend.providers import OpenAIAdapter, AnthropicAdapter

        config = RoutingConfig.from_env()

        # Test case-insensitive lookup
        assert config.get_adapter_class("openai") == OpenAIAdapter
        assert config.get_adapter_class("OPENAI") == OpenAIAdapter
        assert config.get_adapter_class("anthropic") == AnthropicAdapter

        # Test unknown provider
        assert config.get_adapter_class("unknown") is None

    def test_from_env_encryption_key_parameter(self):
        """Test that encryption_key parameter overrides environment."""
        with patch.dict(os.environ, {"ROUTING_ENCRYPTION_KEY": "env_key"}, clear=True):
            config = RoutingConfig.from_env(encryption_key="param_key")
            assert config.encryption_key == "param_key"

    def test_adapter_registry_has_expected_providers(self):
        """Test that all expected providers are in the registry."""
        config = RoutingConfig.from_env()

        expected_providers = [
            "openai",
            "anthropic",
            "grok",
            "deepseek",
            "openrouter",
            "ollama_gcp",
            "llamacpp_gcp",
            "tinylama",
            "siliconeflow",
            "moonshot",
            "elevenlabs",
        ]

        for provider in expected_providers:
            assert provider in config.adapter_registry, f"Missing provider: {provider}"


class TestGetDefaultConfig:
    """Test suite for get_default_config singleton."""

    @patch.dict(os.environ, {"ROUTING_ENCRYPTION_KEY": "test_key"}, clear=True)
    def test_get_default_config_singleton(self):
        """Test that get_default_config returns same instance."""
        # Clear any cached instance
        import backend.services.routing_config as rc

        rc._default_config = None

        config1 = get_default_config()
        config2 = get_default_config()

        # Should return same instance
        assert config1 is config2

    @patch.dict(os.environ, {"ROUTING_ENCRYPTION_KEY": "test_key"}, clear=True)
    def test_get_default_config_validates(self):
        """Test that get_default_config validates configuration."""
        # Clear any cached instance
        import backend.services.routing_config as rc

        rc._default_config = None

        # Should not raise exception
        config = get_default_config()
        assert config is not None
        assert config.adapter_registry  # Non-empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
