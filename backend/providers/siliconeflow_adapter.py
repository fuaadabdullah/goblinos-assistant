"""
SiliconeFlow provider adapter for health checks and model discovery.
SiliconeFlow exposes an OpenAI-compatible API.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class SiliconeflowAdapter:
    """Adapter for SiliconeFlow API provider operations."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.siliconflow.com/v1"
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    async def health_check(self) -> Dict[str, Any]:
        start_time = time.time()
        try:
            models_response = await asyncio.get_event_loop().run_in_executor(
                None, self.client.models.list
            )
            response_time = (time.time() - start_time) * 1000
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
            logger.error("SiliconeFlow health check failed: %s", e)
            return {
                "healthy": False,
                "response_time_ms": round(response_time, 2),
                "error_rate": 1.0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            models_response = await asyncio.get_event_loop().run_in_executor(
                None, self.client.models.list
            )
            models: List[Dict[str, Any]] = []
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
            logger.error("Failed to list SiliconeFlow models: %s", e)
            return self._get_fallback_models()

    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "Qwen/Qwen2.5-7B-Instruct",
                "name": "Qwen 2.5 7B Instruct",
                "capabilities": ["chat"],
                "context_window": 32768,
                "pricing": {"input": 0.00035, "output": 0.00035},
            },
            {
                "id": "Qwen/Qwen2.5-72B-Instruct",
                "name": "Qwen 2.5 72B Instruct",
                "capabilities": ["chat"],
                "context_window": 32768,
                "pricing": {"input": 0.00056, "output": 0.00056},
            },
            {
                "id": "deepseek-ai/DeepSeek-V3",
                "name": "DeepSeek V3",
                "capabilities": ["chat", "code"],
                "context_window": 64000,
                "pricing": {"input": 0.00027, "output": 0.00110},
            },
            {
                "id": "meta-llama/Llama-3.1-8B-Instruct",
                "name": "Llama 3.1 8B Instruct",
                "capabilities": ["chat"],
                "context_window": 128000,
                "pricing": {"input": 0.00035, "output": 0.00035},
            },
            {
                "id": "meta-llama/Llama-3.1-70B-Instruct",
                "name": "Llama 3.1 70B Instruct",
                "capabilities": ["chat"],
                "context_window": 128000,
                "pricing": {"input": 0.00056, "output": 0.00056},
            },
            {
                "id": "THUDM/glm-4-9b-chat",
                "name": "GLM-4 9B Chat",
                "capabilities": ["chat"],
                "context_window": 8192,
                "pricing": {"input": 0.00035, "output": 0.00035},
            },
        ]

    def _infer_capabilities(self, model_id: str) -> List[str]:
        capabilities = ["chat"]
        if "deepseek" in model_id.lower() or "coder" in model_id.lower():
            capabilities.append("code")
        if "vision" in model_id.lower() or "vl" in model_id.lower():
            capabilities.append("vision")
        return capabilities

    def _get_context_window(self, model_id: str) -> int:
        if "Qwen2.5" in model_id:
            return 32768
        if "DeepSeek-V3" in model_id:
            return 64000
        if "Llama-3.1" in model_id:
            return 128000
        if "glm-4" in model_id:
            return 8192
        return 32768

    def _get_pricing(self, model_id: str) -> Dict[str, float]:
        if "7B" in model_id or "8B" in model_id or "9B" in model_id:
            return {"input": 0.00035, "output": 0.00035}
        if "70B" in model_id or "72B" in model_id:
            return {"input": 0.00056, "output": 0.00056}
        if "DeepSeek-V3" in model_id:
            return {"input": 0.00027, "output": 0.00110}
        return {"input": 0.0004, "output": 0.0004}

    async def test_completion(
        self, model: str = "Qwen/Qwen2.5-7B-Instruct", max_tokens: int = 10
    ) -> Dict[str, Any]:
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
            logger.error("SiliconeFlow completion test failed: %s", e)
            return {
                "success": False,
                "response_time_ms": round(response_time, 2),
                "error": str(e),
                "model": model,
            }
