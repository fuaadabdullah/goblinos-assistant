"""
Anthropic provider adapter for health checks and model discovery.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
import anthropic
import logging

from .base_adapter import AdapterBase
from .provider_registry import get_provider_registry

logger = logging.getLogger(__name__)


class AnthropicAdapter(AdapterBase):
    """Adapter for Anthropic API provider operations."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize Anthropic adapter.

        Args:
            api_key: Anthropic API key (optional, will use registry)
            base_url: Optional custom base URL
        """
        registry = get_provider_registry()
        config = registry.get_provider_config_dict("anthropic")

        if config:
            # Allow explicit overrides (e.g., DB-stored credentials/base_url).
            if api_key is not None:
                config["api_key"] = api_key
            if base_url is not None:
                config["base_url"] = base_url
        else:
            # Fallback to manual config if registry fails
            config = {
                "api_key": api_key,
                "base_url": base_url,
                "timeout": 30,
                "retries": 2,
                "cost_per_token_input": 0.008,
                "cost_per_token_output": 0.024,
                "latency_threshold_ms": 4000,
            }

        super().__init__(name="anthropic", config=config)
        # Only pass base_url if explicitly provided (None is valid for Anthropic default)
        if self.base_url:
            self.client = anthropic.Anthropic(
                api_key=self.api_key, base_url=self.base_url
            )
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Anthropic API.

        Returns:
            Dict containing health status and metrics
        """
        start_time = time.time()
        try:
            # Test with a simple message to check API availability
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hello"}],
                ),
            )

            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            return {
                "healthy": True,
                "response_time_ms": round(response_time, 2),
                "error_rate": 0.0,
                "available_models": len(self._get_available_models()),
                "timestamp": time.time(),
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Anthropic health check failed: {e}")

            return {
                "healthy": False,
                "response_time_ms": round(response_time, 2),
                "error_rate": 1.0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Anthropic models.

        Returns:
            List of model information dictionaries
        """
        try:
            # Anthropic doesn't have a models list endpoint, so we return known models
            models = self._get_available_models()

            result = []
            for model_info in models:
                result.append(
                    {
                        "id": model_info["id"],
                        "name": model_info["name"],
                        "capabilities": model_info["capabilities"],
                        "context_window": model_info["context_window"],
                        "pricing": model_info["pricing"],
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Failed to list Anthropic models: {e}")
            return []

    def _get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available Anthropic models with their specifications.

        Returns:
            List of model specifications
        """
        return [
            {
                "id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "capabilities": ["chat", "vision"],
                "context_window": 200000,
                "pricing": {"input": 0.015, "output": 0.075},  # per 1K tokens
            },
            {
                "id": "claude-3-sonnet-20240229",
                "name": "Claude 3 Sonnet",
                "capabilities": ["chat", "vision"],
                "context_window": 200000,
                "pricing": {"input": 0.003, "output": 0.015},
            },
            {
                "id": "claude-3-haiku-20240307",
                "name": "Claude 3 Haiku",
                "capabilities": ["chat", "vision"],
                "context_window": 200000,
                "pricing": {"input": 0.00025, "output": 0.00125},
            },
            {
                "id": "claude-3-5-sonnet-20240620",
                "name": "Claude 3.5 Sonnet",
                "capabilities": ["chat", "vision"],
                "context_window": 200000,
                "pricing": {"input": 0.003, "output": 0.015},
            },
            {
                "id": "claude-2.1",
                "name": "Claude 2.1",
                "capabilities": ["chat"],
                "context_window": 200000,
                "pricing": {"input": 0.008, "output": 0.024},
            },
            {
                "id": "claude-2.0",
                "name": "Claude 2.0",
                "capabilities": ["chat"],
                "context_window": 100000,
                "pricing": {"input": 0.008, "output": 0.024},
            },
            {
                "id": "claude-instant-1.2",
                "name": "Claude Instant 1.2",
                "capabilities": ["chat"],
                "context_window": 100000,
                "pricing": {"input": 0.0008, "output": 0.0024},
            },
        ]

    async def test_completion(
        self, model: str = "claude-3-haiku-20240307", max_tokens: int = 10
    ) -> Dict[str, Any]:
        """Test completion capability.

        Args:
            model: Model to test
            max_tokens: Maximum tokens for response

        Returns:
            Dict with test results
        """
        start_time = time.time()
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": "Hello, test message"}],
                ),
            )

            response_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "response_time_ms": round(response_time, 2),
                "tokens_used": response.usage.input_tokens
                + response.usage.output_tokens
                if hasattr(response, "usage")
                else None,
                "model": model,
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Anthropic completion test failed: {e}")

            return {
                "success": False,
                "response_time_ms": round(response_time, 2),
                "error": str(e),
                "model": model,
            }

    async def generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Generate completion using Anthropic API.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (model, temperature, max_tokens, etc.)

        Returns:
            Dict containing response data
        """
        # Avoid passing duplicate keyword args (e.g. model both explicit and in **kwargs)
        model = kwargs.pop("model", "claude-3-sonnet-20240229")
        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        def _sync_call():
            payload = {
                "model": model,
                "messages": anthropic_messages,
                **kwargs,
            }
            if system_message:
                payload["system"] = system_message
            return self.client.messages.create(**payload)

        response = await self._call_with_circuit_breaker(_sync_call)

        # Extract usage information for cost logging
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        # Log cost using base adapter method
        self._log_cost(input_tokens, output_tokens)

        return {
            "content": response.content[0].text if response.content else "",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "model": model,
            "finish_reason": response.stop_reason,
        }

    async def a_generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Async generate completion using Anthropic API.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Dict containing response data
        """
        # For Anthropic, async and sync are the same since we use run_in_executor
        return await self.generate(messages, **kwargs)
