"""
OpenAI provider implementing the unified ProviderBase interface.
"""

import time
import logging
from typing import Dict, Any, Optional

from openai import OpenAI
from .base import (
    ProviderBase,
    HealthStatus,
    InferenceRequest,
    InferenceResult,
    emit_metrics,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(ProviderBase):
    """OpenAI provider implementing ProviderBase interface."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        models: Optional[list] = None,
        cost_per_token_input: float = 0.0015,
        cost_per_token_output: float = 0.002,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            base_url: Optional custom base URL
            models: List of supported models
            cost_per_token_input: Cost per input token
            cost_per_token_output: Cost per output token
        """
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # Default models if not specified
        self._models = models or ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        self.cost_per_token_input = cost_per_token_input
        self.cost_per_token_output = cost_per_token_output

        # Model capabilities mapping
        self._model_caps = {
            "gpt-4": {"max_tokens": 8192, "vision": False},
            "gpt-4-turbo": {"max_tokens": 128000, "vision": True},
            "gpt-4-vision": {"max_tokens": 128000, "vision": True},
            "gpt-3.5-turbo": {"max_tokens": 16385, "vision": False},
        }

    @property
    def provider_id(self) -> str:
        return "openai"

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities."""
        max_tokens = {}
        supports_vision = False

        for model in self._models:
            caps = self._model_caps.get(model, {"max_tokens": 4096, "vision": False})
            max_tokens[model] = caps["max_tokens"]
            if caps["vision"]:
                supports_vision = True

        return {
            "models": self._models,
            "max_tokens": max_tokens,
            "supports_streaming": True,
            "supports_vision": supports_vision,
            "cost_per_token_input": self.cost_per_token_input,
            "cost_per_token_output": self.cost_per_token_output,
        }

    def health_check(self) -> HealthStatus:
        """Check OpenAI API health."""
        try:
            # Simple health check - try to list models
            response = self.client.models.list()
            if response.data:
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.DEGRADED
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return HealthStatus.UNHEALTHY

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute inference with OpenAI."""
        start_time = time.time()

        try:
            # Prepare messages
            messages = request.messages

            # Build request parameters
            params = {
                "model": request.model,
                "messages": messages,
            }

            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens
            if request.stream:
                params["stream"] = request.stream

            # Make API call
            response = self.client.chat.completions.create(**params)

            # Handle streaming vs non-streaming
            if request.stream:
                # For streaming, we'd need to handle the stream
                # For now, return a placeholder
                content = "Streaming not yet implemented"
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                finish_reason = "stop"
            else:
                content = response.choices[0].message.content
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                finish_reason = response.choices[0].finish_reason

            latency_ms = int((time.time() - start_time) * 1000)

            result = InferenceResult(
                content=content,
                usage=usage,
                model=request.model,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                success=True,
            )

            # Emit metrics
            emit_metrics(
                provider=self.provider_id,
                model=request.model,
                latency_ms=latency_ms,
                success=True,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )

            return result

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"OpenAI inference failed: {e}")

            # Emit failure metrics
            emit_metrics(
                provider=self.provider_id,
                model=request.model,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

            return InferenceResult(
                content="",
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                model=request.model,
                latency_ms=latency_ms,
                success=False,
                error_message=str(e),
            )

    def estimate_cost(self, request: InferenceRequest) -> float:
        """Estimate cost for OpenAI request."""
        # Rough estimation based on message length
        # In practice, you'd tokenize the messages properly
        input_chars = sum(len(msg.get("content", "")) for msg in request.messages)
        estimated_input_tokens = input_chars // 4  # Rough approximation

        # Estimate output tokens (usually 1/3 to 1/2 of input for chat)
        estimated_output_tokens = estimated_input_tokens // 3

        cost = (
            estimated_input_tokens * self.cost_per_token_input
            + estimated_output_tokens * self.cost_per_token_output
        )

        return cost
