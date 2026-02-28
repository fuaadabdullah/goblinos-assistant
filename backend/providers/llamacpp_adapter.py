"""
Llama.cpp provider adapter for local LLM operations.
"""

import time
from typing import Dict, List, Optional, Any
import httpx
import logging

logger = logging.getLogger(__name__)


class LlamaCppAdapter:
    """Adapter for llama.cpp local LLM provider operations."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialize llama.cpp adapter.

        Args:
            api_key: API key for the local proxy
            base_url: Optional custom base URL (defaults to local proxy)
        """
        self.api_key = api_key
        self.base_url = self._normalize_base_url(base_url or "http://localhost:8002")
        self.client = httpx.AsyncClient(timeout=30.0)

    @staticmethod
    def _normalize_base_url(base_url: Optional[str]) -> str:
        if not base_url:
            return ""
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            normalized = normalized[:-3]
        return normalized

    @staticmethod
    def _extract_prompt(messages: List[Dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return messages[-1].get("content", "") if messages else ""

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
        """Send chat completion request to llama.cpp or compatible proxy."""
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
            **kwargs,
        }

        for path in ("/chat/completions", "/v1/chat/completions"):
            try:
                response = await self.client.post(
                    f"{self.base_url}{path}", json=payload, headers=headers
                )
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        return result["choices"][0]["message"]["content"]
                    if "content" in result:
                        return result["content"]
                elif response.status_code not in (404, 405):
                    response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error(f"llama.cpp chat request failed at {path}: {e}")

        prompt = self._extract_prompt(messages)
        completion_payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        response = await self.client.post(
            f"{self.base_url}/completion", json=completion_payload, headers=headers
        )
        response.raise_for_status()
        result = response.json()
        if "content" in result:
            return result["content"]
        if "response" in result:
            return result["response"]
        raise RuntimeError(f"Unexpected llama.cpp response format: {result}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on llama.cpp via local proxy.

        Returns:
            Dict containing health status and metrics
        """
        start_time = time.time()
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                # Get models to check if llama.cpp is responsive
                models_response = await self.client.get(
                    f"{self.base_url}/models",
                    headers={"x-api-key": self.api_key}
                )

                available_models = 0
                if models_response.status_code == 200:
                    data = models_response.json()
                    available_models = len(data.get("models", {}).get("llamacpp", []))

                return {
                    "healthy": True,
                    "response_time_ms": round(response_time, 2),
                    "error_rate": 0.0,
                    "available_models": available_models,
                    "timestamp": time.time(),
                }
            else:
                return {
                    "healthy": False,
                    "response_time_ms": round(response_time, 2),
                    "error_rate": 1.0,
                    "error": f"HTTP {response.status_code}",
                    "timestamp": time.time(),
                }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"llama.cpp health check failed: {e}")

            return {
                "healthy": False,
                "response_time_ms": round(response_time, 2),
                "error_rate": 1.0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available llama.cpp models via local proxy.

        Returns:
            List of model information dictionaries
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/models",
                headers={"x-api-key": self.api_key}
            )

            if response.status_code != 200:
                logger.error(f"Failed to list llama.cpp models: HTTP {response.status_code}")
                return []

            data = response.json()
            llamacpp_models = data.get("models", {}).get("llamacpp", [])

            models = []
            for model_name in llamacpp_models:
                models.append({
                    "id": model_name,
                    "name": model_name,
                    "capabilities": ["chat"],  # llama.cpp models support chat
                    "context_window": self._get_context_window(model_name),
                    "pricing": {"input": 0.0, "output": 0.0},  # Local models are free
                    "provider": "llamacpp",
                    "local": True
                })

            return models

        except Exception as e:
            logger.error(f"Failed to list llama.cpp models: {e}")
            return []

    def _get_context_window(self, model_name: str) -> int:
        """Get context window size for llama.cpp model.

        Args:
            model_name: llama.cpp model name

        Returns:
            Context window size in tokens
        """
        # Common context windows for quantized models
        # These are typically smaller than cloud models due to memory constraints
        context_windows = {
            "active-model": 4096,  # Default for active model
            "llama-7b-q4": 4096,
            "llama-13b-q4": 4096,
            "llama-30b-q3": 2048,
            "mistral-7b-q5": 8192,
            "codellama-7b-q4": 16384,
            "codellama-13b-q4": 16384,
        }

        # Default to 4096 for unknown models
        return context_windows.get(model_name, 4096)

    async def test_completion(
        self, model: str = "active-model", max_tokens: int = 10
    ) -> Dict[str, Any]:
        """Test completion capability via local proxy.

        Args:
            model: Model to test
            max_tokens: Maximum tokens for response

        Returns:
            Dict with test results
        """
        start_time = time.time()
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello, test message"}],
                "max_tokens": max_tokens
            }

            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"x-api-key": self.api_key}
            )

            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return {
                    "success": True,
                    "response_time_ms": round(response_time, 2),
                    "model": model,
                    "local": True
                }
            else:
                return {
                    "success": False,
                    "response_time_ms": round(response_time, 2),
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "model": model,
                }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"llama.cpp completion test failed: {e}")

            return {
                "success": False,
                "response_time_ms": round(response_time, 2),
                "error": str(e),
                "model": model,
            }
