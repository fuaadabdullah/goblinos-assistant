"""
Grok/xAI provider adapter for health checks and model discovery.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from openai import OpenAI
import logging

from .base_adapter import AdapterBase
from .provider_registry import get_provider_registry

logger = logging.getLogger(__name__)


class GrokAdapter(AdapterBase):
    """Adapter for Grok/xAI API provider operations."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize Grok adapter.

        Args:
            api_key: xAI API key (optional, will use registry config)
            base_url: Optional custom base URL
        """
        registry = get_provider_registry()
        config = registry.get_provider_config_dict("grok")

        if config:
            if api_key is not None:
                config["api_key"] = api_key
            if base_url is not None:
                config["base_url"] = base_url
        else:
            config = {
                "api_key": api_key,
                "base_url": base_url or "https://api.x.ai/v1",
                "timeout": 30,
                "retries": 2,
                "cost_per_token_input": 0.005,
                "cost_per_token_output": 0.015,
                "latency_threshold_ms": 4000,
            }

        super().__init__(name="grok", config=config)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Grok/xAI API.

        Returns:
            Dict containing health status and metrics
        """
        start_time = time.time()
        try:
            # Test with models endpoint
            models_response = await asyncio.get_event_loop().run_in_executor(
                None, self.client.models.list
            )

            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            return {
                "healthy": True,
                "response_time_ms": round(response_time, 2),
                "error_rate": 0.0,
                "available_models": len(models_response.data)
                if hasattr(models_response, "data")
                else 0,
                "timestamp": time.time(),
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Grok health check failed: {e}")

            return {
                "healthy": False,
                "response_time_ms": round(response_time, 2),
                "error_rate": 1.0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Grok models.

        Returns:
            List of model information dictionaries
        """
        try:
            models_response = await asyncio.get_event_loop().run_in_executor(
                None, self.client.models.list
            )

            models = []
            for model in models_response.data:
                models.append(
                    {
                        "id": model.id,
                        "name": model.id,
                        "capabilities": self._infer_capabilities(model.id),
                        "context_window": self._get_context_window(model.id),
                        "pricing": self._get_pricing(model.id),
                    }
                )

            return models

        except Exception as e:
            logger.error(f"Failed to list Grok models: {e}")
            # Fallback to known models if API fails
            return self._get_fallback_models()

    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """Get fallback list of known Grok models.

        Returns:
            List of known model specifications
        """
        return [
            {
                "id": "grok-4-latest",
                "name": "Grok 4 Latest",
                "capabilities": ["chat"],
                "context_window": 128000,
                "pricing": {"input": 0.005, "output": 0.015},  # Estimated pricing
            },
            {
                "id": "grok-beta",
                "name": "Grok Beta",
                "capabilities": ["chat"],
                "context_window": 128000,
                "pricing": {"input": 0.005, "output": 0.015},  # Estimated pricing
            },
            {
                "id": "grok-vision-beta",
                "name": "Grok Vision Beta",
                "capabilities": ["chat", "vision"],
                "context_window": 128000,
                "pricing": {"input": 0.005, "output": 0.015},
            },
        ]

    def _infer_capabilities(self, model_id: str) -> List[str]:
        """Infer capabilities based on model ID.

        Args:
            model_id: Grok model identifier

        Returns:
            List of capability strings
        """
        capabilities = ["chat"]  # All Grok models support chat

        # Vision models
        if "vision" in model_id.lower():
            capabilities.append("vision")

        return capabilities

    def _get_context_window(self, model_id: str) -> int:
        """Get context window size for model.

        Args:
            model_id: Grok model identifier

        Returns:
            Context window size in tokens
        """
        # Grok models typically have large context windows
        context_windows = {
            "grok-4-latest": 128000,
            "grok-beta": 128000,
            "grok-vision-beta": 128000,
        }

        return context_windows.get(model_id, 128000)

    def _get_pricing(self, model_id: str) -> Dict[str, float]:
        """Get pricing information for model.

        Args:
            model_id: Grok model identifier

        Returns:
            Dict with pricing per 1K tokens
        """
        # Estimated pricing (subject to change)
        pricing = {
            "grok-4-latest": {"input": 0.005, "output": 0.015},
            "grok-beta": {"input": 0.005, "output": 0.015},
            "grok-vision-beta": {"input": 0.005, "output": 0.015},
        }

        return pricing.get(model_id, {"input": 0.005, "output": 0.015})

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Send chat completion request to Grok/xAI.

        Args:
            model: Model name to use (e.g., "grok-4-latest", "grok-beta")
            messages: List of message dictionaries with role and content
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API

        Returns:
            The text content of the model's response

        Raises:
            Exception: If the API request fails
        """
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=stream,
                    **kwargs,
                ),
            )

            # Extract content from response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logger.error(f"Unexpected response format: {response}")
                raise Exception("Unexpected response format from Grok")

        except Exception as e:
            logger.error(f"Grok chat request failed: {e}")
            raise

    async def test_completion(
        self, model: str = "grok-beta", max_tokens: int = 10
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
            # Use the new chat method for consistency
            await self.chat(
                model=model,
                messages=[{"role": "user", "content": "Hello, test message"}],
                max_tokens=max_tokens,
            )

            response_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "response_time_ms": round(response_time, 2),
                "model": model,
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Grok completion test failed: {e}")

            return {
                "success": False,
                "response_time_ms": round(response_time, 2),
                "error": str(e),
                "model": model,
            }

    async def generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Generate completion using Grok API.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (model, temperature, max_tokens, etc.)

        Returns:
            Dict containing response data
        """
        model = kwargs.get("model", "grok-beta")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1024)
        top_p = kwargs.get("top_p", 1.0)
        stream = kwargs.get("stream", False)

        def _sync_call():
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
            )

        response = await self._call_with_circuit_breaker(_sync_call)

        content = ""
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content or ""

        # Extract token usage from response
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else (input_tokens + output_tokens)

        # Log cost using provider config
        self._log_cost(input_tokens, output_tokens)

        finish_reason = "stop"
        if response.choices and len(response.choices) > 0:
            finish_reason = response.choices[0].finish_reason or "stop"

        return {
            "content": content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "model": model,
            "finish_reason": finish_reason,
        }

    async def a_generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Async generate completion using Grok API.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Dict containing response data
        """
        # For Grok, async and sync are the same
        return await self.generate(messages, **kwargs)
