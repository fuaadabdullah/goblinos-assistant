"""Shared provider adapter base.

Enforces consistent interfaces, retries, telemetry, cost logging, and
circuit-breaker/bulkhead protection for all providers.
"""

import time
import logging
from typing import Any, AsyncGenerator, Dict, Optional, List
from abc import ABC, abstractmethod
import asyncio
from dataclasses import asdict

from .circuit_breaker import get_circuit_breaker, CircuitBreakerOpen
from .bulkhead import get_bulkhead, BulkheadExceeded
from backend.services.types import ProviderConfig, ProviderResult
from backend.services.provider_chunk import ChunkUsage, ProviderChunk

logger = logging.getLogger("providers")


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    def __init__(
        self, provider: str, message: str, details: Optional[Dict[str, Any]] = None
    ):
        self.provider = provider
        self.message = message
        self.details = details or {}
        super().__init__(f"{provider}: {message}")


class AdapterBase(ABC):
    """Thin base class for all provider adapters.

    Provides circuit breaker protection, telemetry, and cost logging.
    Subclasses handle provider-specific logic.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = ProviderConfig.from_dict(name, config)
        self.api_key = self.config.api_key
        self.base_url = self.config.base_url
        self.timeout = self.config.timeout
        self.max_retries = self.config.retries
        self.cost_per_token_input = self.config.cost_per_token_input
        self.cost_per_token_output = self.config.cost_per_token_output
        self.latency_threshold_ms = self.config.latency_threshold_ms

        # Initialize circuit breaker
        self.circuit_breaker = get_circuit_breaker(
            name=name,
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=3,
            timeout=self.timeout,
        )

        # Initialize bulkhead for concurrent request limiting
        self.bulkhead = get_bulkhead(
            name=name,
            max_concurrent=10,  # Allow 10 concurrent requests per provider
        )

    def _log_cost(self, input_tokens: int, output_tokens: int):
        total_cost = (
            input_tokens * self.cost_per_token_input
            + output_tokens * self.cost_per_token_output
        )

        if total_cost > 0:
            logger.info(
                "Provider cost logged",
                extra={
                    "provider": self.name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": total_cost,
                    "timestamp": time.time(),
                },
            )

    async def _call_with_circuit_breaker(self, func, *args, **kwargs) -> Any:
        """Execute a provider call with circuit breaker + bulkhead protection.

        Retries transient failures up to ``max_retries`` times with
        exponential backoff. Every attempt is bounded by an
        ``asyncio.wait_for`` deadline of ``timeout`` seconds so a hung
        provider cannot stall a worker indefinitely.
        """
        try:
            # Fail fast if the circuit is open.
            self.circuit_breaker.before_call()
        except CircuitBreakerOpen as e:
            raise ProviderError(
                self.name, f"Circuit breaker is open: {e}", {"circuit_state": "open"}
            ) from e

        timeout = float(self.timeout) if self.timeout else 30.0
        max_attempts = max(1, int(self.max_retries or 0) + 1)

        async def _invoke_once() -> Any:
            if asyncio.iscoroutinefunction(func):
                coro = func(*args, **kwargs)
            else:

                async def _run_in_thread() -> Any:
                    return await asyncio.to_thread(func, *args, **kwargs)

                coro = _run_in_thread()
            return await asyncio.wait_for(coro, timeout=timeout)

        last_error: Optional[Exception] = None
        try:
            for attempt in range(max_attempts):
                try:
                    async with self.bulkhead.guard():
                        try:
                            result = await _invoke_once()
                        except Exception:
                            self.circuit_breaker.record_failure()
                            raise
                        self.circuit_breaker.record_success()
                        return result
                except BulkheadExceeded:
                    # Local capacity signal: surface immediately, do not retry.
                    raise
                except Exception as e:  # noqa: BLE001 - retried with backoff below
                    last_error = e
                    if attempt < max_attempts - 1:
                        backoff = min(2.0**attempt, 8.0)
                        logger.debug(
                            "Provider %s attempt %d/%d failed (%s); retrying in %.1fs",
                            self.name,
                            attempt + 1,
                            max_attempts,
                            type(e).__name__,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
        except BulkheadExceeded as e:
            raise ProviderError(
                self.name,
                f"Bulkhead limit exceeded: {e}",
                {"bulkhead": "exceeded"},
            ) from e
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                self.name, str(e), {"original_exception": type(e).__name__}
            ) from e

        # All attempts exhausted.
        raise ProviderError(
            self.name,
            f"Provider call failed after {max_attempts} attempt(s): {last_error}",
            {
                "original_exception": type(last_error).__name__
                if last_error is not None
                else None,
                "attempts": max_attempts,
            },
        ) from last_error

    async def _acall_with_circuit_breaker(self, func, *args, **kwargs) -> Any:
        """Backward-compatible alias for async provider calls."""
        return await self._call_with_circuit_breaker(func, *args, **kwargs)

    def get_status(self) -> Dict[str, Any]:
        circuit_status = self.circuit_breaker.get_status()
        bulkhead_status = self.bulkhead.get_status()

        return {
            "provider": self.name,
            "circuit_breaker": circuit_status,
            "bulkhead": bulkhead_status,
            "config": asdict(self.config),
            "healthy": circuit_status["state"] != "open"
            and bulkhead_status["available_slots"] > 0,
        }

    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate text response from the provider.

        Args:
            messages: List of message dictionaries with role/content
            **kwargs: Additional parameters (model, temperature, max_tokens, etc.)

        Returns:
            Dict containing response data with keys:
            - 'content': Generated text content
            - 'usage': Token usage info (input_tokens, output_tokens, total_tokens)
            - 'model': Model used (optional)
            - 'finish_reason': Completion reason (optional)
        """
        pass

    @abstractmethod
    async def a_generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Async version of generate method.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Dict containing response data (same format as generate)
        """
        pass

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Default chat implementation backed by `a_generate`.

        Providers that implement a more efficient native chat method may
        override this.
        """
        result = await self.a_generate(messages, model=model, **kwargs)
        if isinstance(result, dict):
            return (result.get("content") or "").strip()
        return str(result).strip()

    async def stream_chat_completion(
        self,
        request_data: Dict[str, Any],
    ) -> AsyncGenerator[ProviderChunk, None]:
        """Default streaming fallback for adapters without native streaming."""
        result = await self.generate(
            request_data.get("messages") or [],
            model=request_data.get("model"),
            temperature=request_data.get("temperature"),
            max_tokens=request_data.get("max_tokens"),
            top_p=request_data.get("top_p"),
        )

        usage_raw = (result or {}).get("usage") if isinstance(result, dict) else {}
        usage = ChunkUsage(
            prompt_tokens=int(usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)) or 0),
            completion_tokens=int(
                usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)) or 0
            ),
            total_tokens=int(usage_raw.get("total_tokens", 0) or 0),
        )

        if usage.total_tokens <= 0:
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        yield ProviderChunk(
            content=str((result or {}).get("content", "") if isinstance(result, dict) else result),
            role="assistant",
            finish_reason=(result or {}).get("finish_reason", "stop")
            if isinstance(result, dict)
            else "stop",
            usage=usage,
            cost=(result or {}).get("cost") if isinstance(result, dict) else None,
        )
