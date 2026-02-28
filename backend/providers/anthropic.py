"""
Anthropic provider implementing the unified ProviderBase interface.
"""

import time
import logging
from typing import Dict, Any, Optional

import anthropic
from .base import (
    ProviderBase,
    HealthStatus,
    InferenceRequest,
    InferenceResult,
    emit_metrics,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(ProviderBase):
    """Anthropic provider implementing ProviderBase interface."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        models: Optional[list] = None,
        cost_per_token_input: float = 0.008,
        cost_per_token_output: float = 0.024,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            base_url: Optional custom base URL
            models: List of supported models
            cost_per_token_input: Cost per input token
            cost_per_token_output: Cost per output token
        """
        self.api_key = api_key
        self.base_url = base_url

        # Initialize client
        if base_url:
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

        # Default models if not specified
        self._models = models or ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        self.cost_per_token_input = cost_per_token_input
        self.cost_per_token_output = cost_per_token_output

        # Model capabilities mapping
        self._model_caps = {
            "claude-3-opus": {"max_tokens": 200000, "vision": True},
            "claude-3-sonnet": {"max_tokens": 200000, "vision": True},
            "claude-3-haiku": {"max_tokens": 200000, "vision": True},
        }

    @property
    def provider_id(self) -> str:
        return "anthropic"

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities."""
        max_tokens = {}
        supports_vision = False

        for model in self._models:
            caps = self._model_caps.get(model, {"max_tokens": 200000, "vision": True})
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
        """Check Anthropic API health."""
        try:
            # Simple health check - try to get a minimal response
            # Anthropic doesn't have a models list endpoint, so we'll try a minimal message
            response = self.client.messages.create(
                model="claude-3-haiku",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )
            if response.content:
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.DEGRADED
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
            return HealthStatus.UNHEALTHY

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute inference with Anthropic."""
        start_time = time.time()

        try:
            # Convert messages format if needed
            # Anthropic expects different format than OpenAI
            system_message = None
            anthropic_messages = []

            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    system_message = content
                else:
                    # Convert role names
                    anthropic_role = "user" if role == "user" else "assistant"
                    anthropic_messages.append(
                        {"role": anthropic_role, "content": content}
                    )

            # Build request parameters
            params = {
                "model": request.model,
                "messages": anthropic_messages,
                "max_tokens": request.max_tokens or 4096,
            }

            if system_message:
                params["system"] = system_message
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.stream:
                params["stream"] = request.stream

            # Make API call
            response = self.client.messages.create(**params)

            # Handle streaming vs non-streaming
            if request.stream:
                # For streaming, we'd need to handle the stream
                # For now, return a placeholder
                content = "Streaming not yet implemented"
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                finish_reason = "stop"
            else:
                content = response.content[0].text if response.content else ""
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                }
                finish_reason = response.stop_reason or "stop"

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
            logger.error(f"Anthropic inference failed: {e}")

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
        """Estimate cost for Anthropic request."""
        # Rough estimation based on message length
        # In practice, you'd tokenize the messages properly
        input_chars = sum(len(msg.get("content", "")) for msg in request.messages)
        estimated_input_tokens = input_chars // 4  # Rough approximation

        # Estimate output tokens
        estimated_output_tokens = min(
            request.max_tokens or 4096, estimated_input_tokens // 2
        )

        cost = (
            estimated_input_tokens * self.cost_per_token_input
            + estimated_output_tokens * self.cost_per_token_output
        )

        return cost
