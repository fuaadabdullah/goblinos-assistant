"""
Ollama provider implementing the unified ProviderBase interface.
"""

import time
import requests
import logging
from typing import Dict, Any, Optional

from .base import (
    ProviderBase,
    HealthStatus,
    InferenceRequest,
    InferenceResult,
    emit_metrics,
)

logger = logging.getLogger(__name__)


class OllamaProvider(ProviderBase):
    """Ollama provider implementing ProviderBase interface."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: str = "",
        models: Optional[list] = None,
        cost_per_token_input: float = 0.0,
        cost_per_token_output: float = 0.0,
        provider_id: str = "ollama",
        timeout_seconds: int = 60,
    ):
        """Initialize Ollama provider.

        Args:
            base_url: Ollama server base URL
            models: List of supported models
            cost_per_token_input: Cost per input token (usually 0 for local)
            cost_per_token_output: Cost per output token (usually 0 for local)
        """
        self.base_url = base_url
        self.api_key = api_key or ""
        self._provider_id = provider_id
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.cost_per_token_input = cost_per_token_input
        self.cost_per_token_output = cost_per_token_output

        # Default models if not specified
        self._models = models or ["llama2", "codellama", "mistral"]

        # Model capabilities mapping
        self._model_caps = {
            "llama2": {"max_tokens": 4096, "vision": False},
            "codellama": {"max_tokens": 16384, "vision": False},
            "mistral": {"max_tokens": 8192, "vision": False},
        }

    @property
    def provider_id(self) -> str:
        return self._provider_id

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
        """Check Ollama server health."""
        try:
            # Try to get version/tags endpoint
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                headers["X-API-Key"] = self.api_key
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=5, headers=headers
            )
            if response.status_code == 200:
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.DEGRADED
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama health check failed: {e}")
            return HealthStatus.UNHEALTHY

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute inference with Ollama."""
        start_time = time.time()

        try:
            # Prepare request payload
            payload = {
                "model": request.model,
                "messages": request.messages,
                "stream": request.stream,
            }

            if request.temperature is not None:
                payload["temperature"] = request.temperature
            if request.max_tokens is not None:
                payload["max_tokens"] = request.max_tokens

            # Make API call
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                headers["X-API-Key"] = self.api_key
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,  # Ollama can be slow for local inference
            )
            response.raise_for_status()

            result_data = response.json()

            # Handle streaming vs non-streaming
            if request.stream:
                # For streaming, we'd need to handle the stream
                # For now, return a placeholder
                content = "Streaming not yet implemented"
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                finish_reason = "stop"
            else:
                content = result_data.get("message", {}).get("content", "")
                # Ollama doesn't always provide token usage, estimate if missing
                usage = result_data.get("usage", {})
                if not usage:
                    # Rough estimation
                    input_chars = sum(
                        len(msg.get("content", "")) for msg in request.messages
                    )
                    output_chars = len(content)
                    usage = {
                        "input_tokens": input_chars // 4,
                        "output_tokens": output_chars // 4,
                        "total_tokens": (input_chars + output_chars) // 4,
                    }
                finish_reason = result_data.get("done_reason", "stop")

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
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )

            return result

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Ollama inference failed: {e}")

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
        """Estimate cost for Ollama request (usually 0 for local models)."""
        # Local models are typically free
        return 0.0
