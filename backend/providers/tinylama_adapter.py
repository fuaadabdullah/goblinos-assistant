"""
TinyLlama provider adapter for local LLM operations using transformers.
"""

import time
from typing import Dict, List, Optional, Any
import logging

from .base_adapter import AdapterBase
from .provider_registry import get_provider_registry

logger = logging.getLogger(__name__)


class TinyLlamaAdapter(AdapterBase):
    """Adapter for TinyLlama local LLM operations using transformers."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize TinyLlama adapter.

        Args:
            api_key: Not used for local models
            base_url: Not used for local models
        """
        registry = get_provider_registry()
        config = registry.get_provider_config_dict("tinylama")

        if config:
            if api_key is not None:
                config["api_key"] = api_key
            if base_url is not None:
                config["base_url"] = base_url
        else:
            # Fallback to manual config if registry fails
            config = {
                "api_key": api_key or "",
                "base_url": base_url or "",
                "timeout": 120,
                "retries": 1,
                "cost_per_token_input": 0.0,
                "cost_per_token_output": 0.0,
                "latency_threshold_ms": 30000,
            }

        super().__init__(name="tinylama", config=config)

        self.model = None
        self.tokenizer = None
        self._initialized = False

    async def _ensure_model_loaded(self):
        """Lazy load the TinyLlama model."""
        if self._initialized:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info("Loading TinyLlama model...")

            # Use the smallest TinyLlama model for Fly.io resource constraints
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision for memory efficiency
                device_map="auto",  # Auto device placement
                low_cpu_mem_usage=True,
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._initialized = True
            logger.info("TinyLlama model loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            raise RuntimeError("transformers library not available")
        except Exception as e:
            logger.error(f"Failed to load TinyLlama model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Send chat completion request to TinyLlama."""
        await self._ensure_model_loaded()

        try:
            import torch

            # Convert messages to a single prompt
            prompt = self._messages_to_prompt(messages)

            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part (remove the input prompt)
            if full_response.startswith(prompt):
                response = full_response[len(prompt) :].strip()
            else:
                response = full_response.strip()

            return response

        except Exception as e:
            logger.error(f"TinyLlama chat request failed: {e}")
            raise RuntimeError(f"TinyLlama generation failed: {e}")

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt for TinyLlama."""
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Join with newlines and add final assistant prompt
        full_prompt = "\n".join(prompt_parts)
        full_prompt += "\nAssistant:"

        return full_prompt

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on TinyLlama model."""
        start_time = time.time()

        try:
            await self._ensure_model_loaded()

            # Quick test generation
            test_messages = [{"role": "user", "content": "Hello"}]
            response = await self.chat("tinylama", test_messages, max_tokens=10)

            response_time = (time.time() - start_time) * 1000

            return {
                "healthy": True,
                "response_time_ms": round(response_time, 2),
                "error_rate": 0.0,
                "model_loaded": True,
                "timestamp": time.time(),
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"TinyLlama health check failed: {e}")

            return {
                "healthy": False,
                "response_time_ms": round(response_time, 2),
                "error_rate": 1.0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available TinyLlama models."""
        return [
            {
                "id": "tinylama-1.1b-chat",
                "name": "TinyLlama-1.1B-Chat-v1.0",
                "capabilities": ["chat"],
                "context_window": 2048,
                "pricing": {"input": 0.0, "output": 0.0},
                "provider": "tinylama",
                "local": True,
            }
        ]

    async def test_completion(
        self, model: str = "tinylama-1.1b-chat", max_tokens: int = 20
    ) -> Dict[str, Any]:
        """Test completion capability."""
        start_time = time.time()

        try:
            test_messages = [{"role": "user", "content": "Say hello in one word"}]
            response = await self.chat(model, test_messages, max_tokens=max_tokens)

            response_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "response_time_ms": round(response_time, 2),
                "model": model,
                "response": response,
                "local": True,
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"TinyLlama completion test failed: {e}")

            return {
                "success": False,
                "response_time_ms": round(response_time, 2),
                "error": str(e),
                "model": model,
            }

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate text response from TinyLlama.

        Args:
            messages: List of message dictionaries with role/content
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Dict containing response data
        """
        # Run async method in sync context
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.a_generate(messages, **kwargs))
            return result
        finally:
            loop.close()

    async def a_generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Async generate method for TinyLlama.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Dict containing response data
        """
        await self._ensure_model_loaded()

        try:
            import torch

            # Extract parameters
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 512)
            top_p = kwargs.get("top_p", 1.0)

            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)

            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Count input tokens
            input_tokens = inputs["input_ids"].shape[1]

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part
            if full_response.startswith(prompt):
                content = full_response[len(prompt) :].strip()
            else:
                content = full_response.strip()

            # Count output tokens (approximate)
            output_tokens = len(self.tokenizer.encode(content))

            return {
                "content": content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                "model": "tinylama-1.1b-chat",
                "finish_reason": "stop",
            }

        except Exception as e:
            logger.error(f"TinyLlama generation failed: {e}")
            raise ProviderError(
                "tinylama",
                f"Generation failed: {str(e)}",
                {"model": "tinylama-1.1b-chat"},
            )
