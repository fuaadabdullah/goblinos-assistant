import os
import requests
import json
import asyncio
from typing import List, Dict, Any, Optional
import logging

from .base_adapter import AdapterBase
from .provider_registry import get_provider_registry

logger = logging.getLogger(__name__)


class OllamaAdapter(AdapterBase):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        def _normalize_base_url(url: Optional[str]) -> Optional[str]:
            if not url:
                return url
            normalized = url.rstrip("/")
            if normalized.endswith("/v1"):
                normalized = normalized[:-3]
            return normalized

        registry = get_provider_registry()
        config = registry.get_provider_config_dict("ollama")

        if config:
            if api_key is not None:
                config["api_key"] = api_key
            if base_url is not None:
                config["base_url"] = _normalize_base_url(base_url)
            else:
                config["base_url"] = _normalize_base_url(config.get("base_url"))
        else:
            # Fallback to manual config if registry fails
            if base_url is None:
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            config = {
                "api_key": api_key,
                "base_url": _normalize_base_url(base_url),
                "timeout": 30,
                "retries": 2,
                "cost_per_token_input": 0.0,
                "cost_per_token_output": 0.0,
                "latency_threshold_ms": 10000,
            }

        super().__init__(name="ollama", config=config)

    def get_status(self):
        try:
            response = requests.get(f"{self.base_url}/api/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from the Ollama server."""
        try:
            # Try OpenAI-compatible endpoint first
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                # Some proxies use X-API-Key instead of Authorization; include both
                headers["X-API-Key"] = self.api_key

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(
                    f"{self.base_url}/v1/models", headers=headers, timeout=10
                ),
            )
            response.raise_for_status()
            result = response.json()

            if "data" in result:
                # OpenAI format
                return [
                    {
                        "id": model["id"],
                        "name": model.get("id", ""),
                        "capabilities": ["chat"],
                        "context_window": model.get("context_length", 4096),
                        "pricing": {},
                    }
                    for model in result["data"]
                ]
            else:
                # Try Ollama native endpoint
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: requests.get(f"{self.base_url}/api/tags", timeout=10)
                )
                response.raise_for_status()
                result = response.json()

                if "models" in result:
                    return [
                        {
                            "id": model["name"],
                            "name": model["name"],
                            "capabilities": ["chat"],
                            "context_window": model.get("context_length", 4096),
                            "pricing": {},
                        }
                        for model in result["models"]
                    ]

            # Fallback: return some default models
            return [
                {
                    "id": "llama3.2:3b",
                    "name": "Llama 3.2 3B",
                    "capabilities": ["chat"],
                    "context_window": 4096,
                    "pricing": {},
                },
                {
                    "id": "gemma:2b",
                    "name": "Gemma 2B",
                    "capabilities": ["chat"],
                    "context_window": 8192,
                    "pricing": {},
                },
            ]

        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            # Return default models on error
            return [
                {
                    "id": "llama3.2:3b",
                    "name": "Llama 3.2 3B",
                    "capabilities": ["chat"],
                    "context_window": 4096,
                    "pricing": {},
                },
            ]

    def generate_simple(self, prompt, model="llama2"):
        try:
            payload = {"prompt": prompt, "model": model, "stream": False}
            response = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            text = response.text.strip()
            lines = text.split("\n")
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
            return {
                "status": "error",
                "error": "No valid JSON found",
                "raw": text[:500],
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

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
        """Send chat completion request to Ollama/OpenAI-compatible API.

        Args:
            model: Model name to use
            messages: List of message dictionaries with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            The text content of the model's response
        """
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                headers["X-API-Key"] = self.api_key

            openai_payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": stream,
                **kwargs,
            }

            native_options = {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
            }
            for key, value in kwargs.items():
                if isinstance(value, (str, int, float, bool)):
                    native_options[key] = value

            native_payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": native_options,
            }

            def _extract_content(result: Dict[str, Any]) -> Optional[str]:
                choices = result.get("choices") or []
                if choices:
                    return (choices[0].get("message") or {}).get("content")
                message = result.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if content:
                        return content
                if result.get("content"):
                    return result["content"]
                if result.get("response"):
                    return result["response"]
                return None

            # Native Ollama endpoint first, OpenAI-compatible fallback second.
            for path, payload in (
                ("/api/chat", native_payload),
                ("/v1/chat/completions", openai_payload),
            ):
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda p=path, b=payload: requests.post(
                        f"{self.base_url}{p}",
                        json=b,
                        headers=headers,
                        timeout=self.timeout,
                    ),
                )
                if response.status_code in (404, 405):
                    continue
                response.raise_for_status()
                result = response.json()
                content = _extract_content(result)
                if content:
                    return content

                logger.error(f"Unexpected response format from Ollama at {path}: {result}")

            raise Exception("Unexpected response format from Ollama API")

        except Exception as e:
            logger.error(f"Ollama chat request failed: {e}")
            raise

    async def generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Generate completion using Ollama API.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (model, temperature, max_tokens, etc.)

        Returns:
            Dict containing response data
        """
        model = kwargs.pop("model", "llama2")

        # Use the existing chat method but wrap it to match the interface
        content = await self.chat(model, messages, **kwargs)

        # Ollama doesn't provide token usage info, so we estimate
        # Rough estimation: 4 chars per token
        input_chars = sum(len(msg.get("content", "")) for msg in messages)
        output_chars = len(content)
        input_tokens = input_chars // 4
        output_tokens = output_chars // 4

        # Log cost (Ollama is typically free/local)
        self._log_cost(input_tokens, output_tokens)

        return {
            "content": content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "model": model,
            "finish_reason": "stop",  # Ollama doesn't provide this
        }

    async def a_generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Async generate completion using Ollama API.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Dict containing response data
        """
        # For Ollama, async and sync are the same
        return await self.generate(messages, **kwargs)
