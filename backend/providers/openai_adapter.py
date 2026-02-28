"""
OpenAI provider adapter for health checks and model discovery.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from openai import OpenAI
import logging

from .base_adapter import AdapterBase
from .provider_registry import get_provider_registry
from backend.services.provider_chunk import ChunkUsage, ProviderChunk

logger = logging.getLogger(__name__)


class OpenAIAdapter(AdapterBase):
    """Adapter for OpenAI API provider operations."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key (optional, will use registry)
            base_url: Optional custom base URL
        """
        registry = get_provider_registry()
        config = registry.get_provider_config_dict("openai")

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
                "base_url": base_url or "https://api.openai.com/v1",
                "timeout": 30,
                "retries": 2,
                "cost_per_token_input": 0.0015,
                "cost_per_token_output": 0.002,
                "latency_threshold_ms": 3000,
            }

        super().__init__(name="openai", config=config)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on OpenAI API.

        Returns:
            Dict containing health status and metrics
        """
        start_time = time.time()
        try:
            # Simple health check using models endpoint
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
            logger.error(f"OpenAI health check failed: {e}")

            return {
                "healthy": False,
                "response_time_ms": round(response_time, 2),
                "error_rate": 1.0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available OpenAI models.

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
            logger.error(f"Failed to list OpenAI models: {e}")
            return []

    def _infer_capabilities(self, model_id: str) -> List[str]:
        """Infer capabilities based on model ID.

        Args:
            model_id: OpenAI model identifier

        Returns:
            List of capability strings
        """
        capabilities = ["chat"]  # All OpenAI models support chat

        # GPT-4 models support vision
        if "gpt-4" in model_id and (
            "vision" in model_id or model_id.startswith("gpt-4")
        ):
            capabilities.append("vision")

        # GPT-3.5 and GPT-4 support embeddings (though dedicated embedding models exist)
        if "gpt" in model_id:
            capabilities.append("embeddings")

        # Add function calling for newer models
        if any(x in model_id for x in ["gpt-4", "gpt-3.5-turbo"]):
            capabilities.append("functions")

        return capabilities

    def _get_context_window(self, model_id: str) -> int:
        """Get context window size for model.

        Args:
            model_id: OpenAI model identifier

        Returns:
            Context window size in tokens
        """
        # Context windows based on OpenAI documentation
        context_windows = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-1106-preview": 128000,
            "gpt-4-vision-preview": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-3.5-turbo-1106": 16384,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "text-curie-001": 2049,
            "text-babbage-001": 2049,
            "text-ada-001": 2049,
        }

        # Default to 4096 for unknown models
        return context_windows.get(model_id, 4096)

    def _get_pricing(self, model_id: str) -> Dict[str, float]:
        """Get pricing information for model.

        Args:
            model_id: OpenAI model identifier

        Returns:
            Dict with pricing per 1K tokens
        """
        # Pricing in USD per 1K tokens (approximate, subject to change)
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
            "gpt-4-vision-preview": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
        }

        return pricing.get(model_id, {"input": 0.002, "output": 0.002})

    async def test_completion(
        self, model: str = "gpt-3.5-turbo", max_tokens: int = 10
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
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello, test message"}],
                    max_tokens=max_tokens,
                ),
            )

            response_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "response_time_ms": round(response_time, 2),
                "tokens_used": response.usage.total_tokens
                if hasattr(response, "usage")
                else None,
                "model": model,
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"OpenAI completion test failed: {e}")

            return {
                "success": False,
                "response_time_ms": round(response_time, 2),
                "error": str(e),
                "model": model,
            }

    async def generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Generate completion using OpenAI API.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (model, temperature, max_tokens, etc.)

        Returns:
            Dict containing response data
        """
        # Avoid passing duplicate keyword args (e.g. model both explicit and in **kwargs)
        model = kwargs.pop("model", "gpt-3.5-turbo")

        def _sync_call():
            return self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )

        response = await self._call_with_circuit_breaker(_sync_call)

        # Extract usage information for cost logging
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        # Log cost using base adapter method
        self._log_cost(input_tokens, output_tokens)

        return {
            "content": response.choices[0].message.content if response.choices else "",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": (getattr(usage, "total_tokens", 0) if usage else 0),
            },
            "model": model,
            "finish_reason": response.choices[0].finish_reason
            if response.choices
            else None,
        }

    async def a_generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Async generate completion using OpenAI API.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Dict containing response data
        """
        # For OpenAI, async and sync are the same since we use run_in_executor
        return await self.generate(messages, **kwargs)

    async def stream_chat_completion(
        self,
        request_data: Dict[str, Any],
    ):
        """Stream completion chunks using OpenAI's streaming API."""
        messages = request_data.get("messages") or []
        model = request_data.get("model") or "gpt-3.5-turbo"

        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if request_data.get("temperature") is not None:
            params["temperature"] = request_data["temperature"]
        if request_data.get("max_tokens") is not None:
            params["max_tokens"] = request_data["max_tokens"]
        if request_data.get("top_p") is not None:
            params["top_p"] = request_data["top_p"]

        def _sync_open_stream():
            return self.client.chat.completions.create(**params)

        stream = await self._call_with_circuit_breaker(_sync_open_stream)

        completion_tokens = 0
        prompt_tokens = sum(max(1, len((m.get("content") or "")) // 4) for m in messages)

        for raw_chunk in stream:
            if not getattr(raw_chunk, "choices", None):
                continue
            delta = raw_chunk.choices[0].delta
            content = getattr(delta, "content", None) or ""
            finish_reason = getattr(raw_chunk.choices[0], "finish_reason", None)
            if not content and not finish_reason:
                continue
            if content:
                completion_tokens += max(1, len(content) // 4)
            yield ProviderChunk(
                content=content,
                role=getattr(delta, "role", "assistant") or "assistant",
                finish_reason=finish_reason,
            )
            await asyncio.sleep(0)

        usage = ChunkUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        cost = (
            usage.prompt_tokens * self.cost_per_token_input
            + usage.completion_tokens * self.cost_per_token_output
        )
        self._log_cost(usage.prompt_tokens, usage.completion_tokens)
        yield ProviderChunk(content="", role="assistant", finish_reason="stop", usage=usage, cost=cost)
