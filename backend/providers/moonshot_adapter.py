"""
Moonshot AI provider adapter for health checks and model discovery.
Moonshot AI (Kimi) is OpenAI-compatible API.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class MoonshotAdapter:
    """Adapter for Moonshot AI (Kimi) API provider operations."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialize Moonshot adapter.

        Args:
            api_key: Moonshot AI API key
            base_url: Optional custom base URL
        """
        self.api_key = api_key
        # Correct endpoint is api.moonshot.ai (not .cn)
        self.base_url = base_url or "https://api.moonshot.ai/v1"
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Moonshot AI API.

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
            logger.error(f"Moonshot health check failed: {e}")

            return {
                "healthy": False,
                "response_time_ms": round(response_time, 2),
                "error_rate": 1.0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Moonshot AI models.

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
            logger.error(f"Failed to list Moonshot models: {e}")
            # Fallback to known models if API fails
            return self._get_fallback_models()

    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """Get fallback list of known Moonshot AI models.

        Returns:
            List of known model specifications
        """
        return [
            {
                "id": "kimi-k2-turbo-preview",
                "name": "Kimi K2 Turbo Preview",
                "capabilities": ["chat"],
                "context_window": 200000,
                "pricing": {"input": 0.002, "output": 0.002},  # CNY per 1K tokens
            },
            {
                "id": "kimi-k2-thinking-turbo",
                "name": "Kimi K2 Thinking Turbo",
                "capabilities": ["chat", "reasoning"],
                "context_window": 200000,
                "pricing": {"input": 0.003, "output": 0.003},
            },
            {
                "id": "kimi-k2-thinking",
                "name": "Kimi K2 Thinking",
                "capabilities": ["chat", "reasoning"],
                "context_window": 200000,
                "pricing": {"input": 0.003, "output": 0.003},
            },
            {
                "id": "kimi-latest",
                "name": "Kimi Latest",
                "capabilities": ["chat"],
                "context_window": 200000,
                "pricing": {"input": 0.002, "output": 0.002},
            },
            {
                "id": "moonshot-v1-8k",
                "name": "Moonshot v1 8K",
                "capabilities": ["chat"],
                "context_window": 8000,
                "pricing": {"input": 0.012, "output": 0.012},
            },
            {
                "id": "moonshot-v1-32k",
                "name": "Moonshot v1 32K",
                "capabilities": ["chat"],
                "context_window": 32000,
                "pricing": {"input": 0.024, "output": 0.024},
            },
            {
                "id": "moonshot-v1-128k",
                "name": "Moonshot v1 128K",
                "capabilities": ["chat"],
                "context_window": 128000,
                "pricing": {"input": 0.06, "output": 0.06},
            },
            {
                "id": "moonshot-v1-auto",
                "name": "Moonshot v1 Auto",
                "capabilities": ["chat"],
                "context_window": 128000,
                "pricing": {"input": 0.06, "output": 0.06},
            },
        ]

    def _infer_capabilities(self, model_id: str) -> List[str]:
        """Infer capabilities based on model ID.

        Args:
            model_id: Moonshot model identifier

        Returns:
            List of capability strings
        """
        capabilities = ["chat"]  # All models support chat

        # Thinking/reasoning models
        if "thinking" in model_id.lower() or "k2" in model_id.lower():
            capabilities.append("reasoning")

        return capabilities

    def _get_context_window(self, model_id: str) -> int:
        """Get context window size for model.

        Args:
            model_id: Moonshot model identifier

        Returns:
            Context window size in tokens
        """
        # Known context windows
        if "8k" in model_id.lower():
            return 8000
        elif "32k" in model_id.lower():
            return 32000
        elif "128k" in model_id.lower():
            return 128000
        elif "v2" in model_id.lower():
            return 200000

        # Default
        return 8000

    def _get_pricing(self, model_id: str) -> Dict[str, float]:
        """Get pricing information for model.

        Args:
            model_id: Moonshot model identifier

        Returns:
            Dict with pricing per 1K tokens (in CNY)
        """
        # Moonshot AI pricing (in Chinese Yuan per 1K tokens)
        if "v2" in model_id.lower():
            if "thinking" in model_id.lower():
                return {"input": 0.003, "output": 0.003}
            return {"input": 0.002, "output": 0.002}
        elif "128k" in model_id.lower():
            return {"input": 0.06, "output": 0.06}
        elif "32k" in model_id.lower():
            return {"input": 0.024, "output": 0.024}
        elif "8k" in model_id.lower():
            return {"input": 0.012, "output": 0.012}

        # Default pricing
        return {"input": 0.012, "output": 0.012}

    async def test_completion(
        self, model: str = "kimi-k2-turbo-preview", max_tokens: int = 50
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
            logger.error(f"Moonshot completion test failed: {e}")

            return {
                "success": False,
                "response_time_ms": round(response_time, 2),
                "error": str(e),
                "model": model,
            }
