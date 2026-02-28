"""
Google Gemini provider adapter for health checks and model discovery.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
import google.generativeai as genai
import logging

from .base_adapter import AdapterBase
from .provider_registry import get_provider_registry

logger = logging.getLogger(__name__)


class GeminiAdapter(AdapterBase):
    """Adapter for Google Gemini API provider operations."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize Gemini adapter.

        Args:
            api_key: Google AI API key (optional, will use registry config)
            base_url: Optional custom base URL (not used for Gemini)
        """
        registry = get_provider_registry()
        config = registry.get_provider_config_dict("gemini")

        if config:
            if api_key is not None:
                config["api_key"] = api_key
            if base_url is not None:
                config["base_url"] = base_url
        else:
            config = {
                "api_key": api_key,
                "base_url": base_url or "",
                "timeout": 30,
                "retries": 2,
                "cost_per_token_input": 0.00025,
                "cost_per_token_output": 0.0005,
                "latency_threshold_ms": 4000,
            }

        super().__init__(name="gemini", config=config)
        genai.configure(api_key=self.api_key)
        self.client = genai

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Google Gemini API.

        Returns:
            Dict containing health status and metrics
        """
        start_time = time.time()
        try:
            # Test with a simple generation to check API availability
            model = self.client.GenerativeModel("gemini-pro")
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(
                    "Hello", generation_config={"max_output_tokens": 10}
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
            logger.error(f"Gemini health check failed: {e}")

            return {
                "healthy": False,
                "response_time_ms": round(response_time, 2),
                "error_rate": 1.0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Google Gemini models.

        Returns:
            List of model information dictionaries
        """
        try:
            # Return known Gemini models since the API doesn't provide a comprehensive list
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
            logger.error(f"Failed to list Gemini models: {e}")
            return []

    def _get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available Gemini models with their specifications.

        Returns:
            List of model specifications
        """
        return [
            {
                "id": "gemini-pro",
                "name": "Gemini Pro",
                "capabilities": ["chat"],
                "context_window": 32768,
                "pricing": {"input": 0.00025, "output": 0.0005},  # per 1K characters
            },
            {
                "id": "gemini-pro-vision",
                "name": "Gemini Pro Vision",
                "capabilities": ["chat", "vision"],
                "context_window": 16384,
                "pricing": {"input": 0.00025, "output": 0.0005},
            },
            {
                "id": "gemini-1.5-pro",
                "name": "Gemini 1.5 Pro",
                "capabilities": ["chat", "vision"],
                "context_window": 2097152,  # 2M tokens
                "pricing": {"input": 0.00125, "output": 0.005},
            },
            {
                "id": "gemini-1.5-flash",
                "name": "Gemini 1.5 Flash",
                "capabilities": ["chat", "vision"],
                "context_window": 1048576,  # 1M tokens
                "pricing": {"input": 0.000075, "output": 0.0003},
            },
            {
                "id": "gemini-1.0-pro",
                "name": "Gemini 1.0 Pro",
                "capabilities": ["chat"],
                "context_window": 32768,
                "pricing": {"input": 0.00025, "output": 0.0005},
            },
        ]

    async def test_completion(
        self, model: str = "gemini-pro", max_tokens: int = 10
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
            genai_model = self.client.GenerativeModel(model)
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: genai_model.generate_content(
                    "Hello, test message",
                    generation_config={"max_output_tokens": max_tokens},
                ),
            )

            response_time = (time.time() - start_time) * 1000

            # Check if response was successful
            # Note: Gemini may return empty parts with finish_reason MAX_TOKENS or SAFETY
            # which still means the API call succeeded
            if response and response.candidates:
                success = True
            else:
                success = False

            return {
                "success": success,
                "response_time_ms": round(response_time, 2),
                "tokens_used": None,  # Gemini doesn't provide token counts in the same way
                "model": model,
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Gemini completion test failed: {e}")

            return {
                "success": False,
                "response_time_ms": round(response_time, 2),
                "error": str(e),
                "model": model,
            }

    async def generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Generate completion using Gemini API.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (model, temperature, max_tokens, etc.)

        Returns:
            Dict containing response data
        """
        model = kwargs.get("model", "gemini-pro")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1024)

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Gemini uses 'model' for assistant responses
            if role == "assistant":
                role = "model"
            elif role not in ["user", "model"]:
                role = "user"  # Default to user for unknown roles

            gemini_messages.append({"role": role, "parts": [content]})

        def _sync_call():
            genai_model = self.client.GenerativeModel(model)
            chat = genai_model.start_chat(
                history=gemini_messages[:-1]
            )  # All but last message

            return chat.send_message(
                gemini_messages[-1]["parts"][0],
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )

        response = await self._call_with_circuit_breaker(_sync_call)

        content = response.text if response.text else ""

        # Gemini doesn't provide detailed token usage, so we estimate
        # Rough estimation: 4 chars per token
        input_chars = sum(len(msg.get("content", "")) for msg in messages)
        output_chars = len(content)
        input_tokens = input_chars // 4
        output_tokens = output_chars // 4

        # Log cost using provider config
        self._log_cost(input_tokens, output_tokens)

        return {
            "content": content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "model": model,
            "finish_reason": "stop",  # Gemini doesn't provide detailed finish reasons
        }

    async def a_generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Async generate completion using Gemini API.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Dict containing response data
        """
        # For Gemini, async and sync are the same
        return await self.generate(messages, **kwargs)
