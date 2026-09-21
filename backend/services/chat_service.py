from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import Request
from pydantic import BaseModel

from ..errors import map_exception_to_problem
from ..providers.base_adapter import ProviderError
from .chat_context import ChatContext, build_generate_context
from .imports import get_gateway_service
from .provider_chunk import ProviderChunk
from .request_validation import ChatCompletionRequest, ChatMessage
from .response_builder import ChatCompletionResponse
from .routing_provider_selector import RoutingProviderSelector
from .token_accounting import TokenAccountingService

logger = logging.getLogger(__name__)

# Bounded total time for the whole provider-failover loop (selection +
# execution attempts across providers). Individual provider calls also carry
# their own per-call timeouts (see providers/base_adapter.py); this deadline
# backstops the loop as a whole.
def _failover_total_timeout_seconds() -> float:
    try:
        return max(
            1.0, float(os.getenv("CHAT_FAILOVER_TOTAL_TIMEOUT_SECONDS", "120"))
        )
    except (TypeError, ValueError):
        return 120.0


FAILOVER_TOTAL_TIMEOUT_SECONDS = _failover_total_timeout_seconds()


class StreamChatMessage(BaseModel):
    role: str
    content: str


class StreamChatRequest(BaseModel):
    messages: list[StreamChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class ChatService:
    """Authoritative chat execution service for both sync and streaming transport."""

    def __init__(self, routing_service: Any, gateway_service: Any | None = None):
        self._routing_service = routing_service
        self._gateway_service = gateway_service or get_gateway_service()
        self._selector = RoutingProviderSelector(routing_service)
        self._token_accountant = TokenAccountingService()

    async def complete(
        self,
        request: ChatCompletionRequest,
        http_request: Request,
    ) -> ChatCompletionResponse:
        correlation_id = getattr(http_request.state, "correlation_id", None)
        context = await self._build_context(request, correlation_id)

        failures: list[dict[str, str]] = []
        try:
            completion, routed_provider, routing_result = await asyncio.wait_for(
                self._complete_with_failover(
                    request=request,
                    http_request=http_request,
                    context=context,
                    failures=failures,
                ),
                timeout=FAILOVER_TOTAL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderError(
                "failover",
                "Chat completion exceeded the "
                f"{FAILOVER_TOTAL_TIMEOUT_SECONDS:g}s failover deadline",
                {"failures": failures},
            ) from exc

        response_id = routed_provider.request_id or correlation_id or str(uuid.uuid4())
        created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        usage = completion.get("usage") or {}

        return ChatCompletionResponse(
            id=response_id,
            object="chat.completion",
            created=created,
            model=str(completion.get("model") or routed_provider.model),
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(completion.get("content", "") or ""),
                    },
                    "finish_reason": completion.get("finish_reason") or "stop",
                }
            ],
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            },
            metadata={
                "provider": routed_provider.name,
                "routing": routing_result,
            },
        )

    async def _complete_with_failover(
        self,
        request: ChatCompletionRequest,
        http_request: Request,
        context: ChatContext,
        failures: list[dict[str, str]],
    ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        """Execute a chat completion, failing over across ranked providers.

        Each attempt re-runs intelligent selection with previously-failed
        providers excluded, so the next-best candidate is chosen by the
        untouched scoring logic. Raises ProviderError when no providers
        remain.
        """
        excluded: set[str] = set()
        routing_result: dict[str, Any] = {}

        while True:
            try:
                routed_provider, routing_result = (
                    await self._selector.select_for_request(
                        request=request,
                        http_request=http_request,
                        messages=context.messages,
                        gateway_result=context.gateway_result,
                        exclude_providers=excluded or None,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - selection failure
                if not excluded:
                    # First attempt: preserve the original error surface
                    # (e.g. routing 503 / rate-limit responses) exactly.
                    raise
                raise ProviderError(
                    "failover",
                    "All providers failed",
                    {
                        "failures": failures,
                        "selection_error": f"{type(exc).__name__}: {exc}",
                    },
                ) from exc

            if routed_provider.name in excluded:
                # Selection ignored the exclusion list; no progress possible.
                raise ProviderError(
                    "failover",
                    "All providers failed",
                    {"failures": failures},
                )

            request_data = self._build_provider_request_data(
                request=request,
                messages=context.messages,
                selected_model=routed_provider.model,
            )
            try:
                completion = await routed_provider.chat_completion(request_data)
            except Exception as exc:  # noqa: BLE001 - provider failure: fail over
                failures.append(
                    {
                        "provider": routed_provider.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                logger.warning(
                    "Chat provider '%s' failed (%s); failing over to next provider",
                    routed_provider.name,
                    type(exc).__name__,
                )
                await self._record_provider_failure(
                    routing_result, routed_provider, exc
                )
                excluded.add(routed_provider.name)
                continue

            return completion, routed_provider, routing_result

    async def _record_provider_failure(
        self,
        routing_result: dict[str, Any],
        routed_provider: Any,
        exc: Exception,
    ) -> None:
        """Best-effort: record an execution failure for the routing scorer.

        Never raises; failure telemetry must not break the chat path.
        """
        try:
            recorder = getattr(self._routing_service, "_log_routing_request", None)
            if not callable(recorder):
                return
            provider_info = getattr(routed_provider, "provider_info", None) or {}
            await recorder(
                request_id=str(uuid.uuid4()),
                capability="chat",
                requirements=routing_result.get("requirements") or {},
                selected_provider_id=provider_info.get("id"),
                success=False,
                error_message=f"{type(exc).__name__}: {exc}",
            )
        except Exception:  # noqa: BLE001 - telemetry is best-effort
            logger.debug("Failed to record provider failure", exc_info=True)

    async def stream(
        self,
        request: ChatCompletionRequest,
        http_request: Request,
    ) -> AsyncGenerator[str, None]:
        correlation_id = getattr(http_request.state, "correlation_id", None)
        context = await self._build_context(request, correlation_id)

        # Fail over across providers until the first chunk is produced.
        # Once chunks have been yielded the response is committed, so a
        # mid-stream failure is surfaced as an error chunk (as before).
        excluded: set[str] = set()
        failures: list[dict[str, str]] = []
        deadline = time.monotonic() + FAILOVER_TOTAL_TIMEOUT_SECONDS
        error_model = request.model or "unknown"

        while True:
            if excluded and time.monotonic() >= deadline:
                break
            try:
                routed_provider, routing_result = (
                    await self._selector.select_for_request(
                        request=request,
                        http_request=http_request,
                        messages=context.messages,
                        gateway_result=context.gateway_result,
                        exclude_providers=excluded or None,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - selection failure
                failures.append(
                    {
                        "provider": "<selection>",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                break
            if routed_provider.name in excluded:
                # Selection ignored the exclusion list; no progress possible.
                break

            request_data = self._build_provider_request_data(
                request=request,
                messages=context.messages,
                selected_model=routed_provider.model,
            )
            error_model = routed_provider.model
            chunk_id = routed_provider.request_id or correlation_id or str(uuid.uuid4())
            streamed_any = False
            try:
                async for chunk in routed_provider.stream_chat_completion(request_data):
                    streamed_any = True
                    payload = self._format_openai_chunk(
                        request_id=chunk_id,
                        model=routed_provider.model,
                        chunk=chunk,
                    )
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
                return
            except Exception as exc:  # noqa: BLE001
                if streamed_any:
                    # Mid-stream failure: chunks already went out; cannot fail over.
                    yield self._format_openai_error_chunk(
                        chunk_id, routed_provider.model, exc, correlation_id
                    )
                    yield "data: [DONE]\n\n"
                    return
                failures.append(
                    {
                        "provider": routed_provider.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                logger.warning(
                    "Streaming provider '%s' failed before first chunk (%s); "
                    "failing over to next provider",
                    routed_provider.name,
                    type(exc).__name__,
                )
                await self._record_provider_failure(
                    routing_result, routed_provider, exc
                )
                excluded.add(routed_provider.name)
                continue

        # Every provider failed before producing a chunk.
        problem = map_exception_to_problem(
            ProviderError("failover", "All providers failed", {"failures": failures}),
            correlation_id,
        )
        error_chunk = {
            "id": correlation_id or str(uuid.uuid4()),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": error_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                    },
                    "finish_reason": "error",
                }
            ],
            "error": problem.model_dump(),
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    async def stream_legacy(
        self,
        request: ChatCompletionRequest,
        http_request: Request,
    ) -> AsyncGenerator[str, None]:
        correlation_id = getattr(http_request.state, "correlation_id", None)
        context = await self._build_context(request, correlation_id)

        # Fail over across providers until the first content chunk is
        # produced. A fresh 'started' event is emitted per attempt so the
        # client always knows which provider is serving the stream.
        excluded: set[str] = set()
        failures: list[dict[str, str]] = []
        deadline = time.monotonic() + FAILOVER_TOTAL_TIMEOUT_SECONDS

        while True:
            if excluded and time.monotonic() >= deadline:
                break
            try:
                routed_provider, routing_result = (
                    await self._selector.select_for_request(
                        request=request,
                        http_request=http_request,
                        messages=context.messages,
                        gateway_result=context.gateway_result,
                        exclude_providers=excluded or None,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - selection failure
                failures.append(
                    {
                        "provider": "<selection>",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                break
            if routed_provider.name in excluded:
                break

            request_data = self._build_provider_request_data(
                request=request,
                messages=context.messages,
                selected_model=routed_provider.model,
            )

            total_tokens = 0
            final_cost = 0.0
            started_at = time.monotonic()

            yield (
                "data: "
                f"{json.dumps({'status': 'started', 'provider': routed_provider.name, 'model': routed_provider.model})}"
                "\n\n"
            )

            streamed_any = False
            try:
                async for chunk in routed_provider.stream_chat_completion(request_data):
                    if chunk.content:
                        streamed_any = True
                        tok_count = self._token_accountant.count_tokens(chunk.content)
                        total_tokens += tok_count
                        yield (
                            "data: "
                            f"{json.dumps({'content': chunk.content, 'token_count': tok_count, 'done': False})}"
                            "\n\n"
                        )

                    if chunk.usage and chunk.usage.total_tokens:
                        total_tokens = chunk.usage.total_tokens
                    if chunk.cost is not None:
                        final_cost = chunk.cost

                duration_ms = int((time.monotonic() - started_at) * 1000)
                completion_payload = {
                    "tokens": total_tokens,
                    "cost": round(final_cost, 6),
                    "model": routed_provider.model,
                    "provider": routed_provider.name,
                    "duration_ms": duration_ms,
                    "done": True,
                }
                yield f"data: {json.dumps(completion_payload)}\n\n"
                return
            except Exception as exc:  # noqa: BLE001
                if streamed_any:
                    # Mid-stream failure: content already went out; cannot fail over.
                    logger.error("Legacy streaming failed: %s", exc)
                    yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"
                    return
                failures.append(
                    {
                        "provider": routed_provider.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                logger.warning(
                    "Legacy streaming provider '%s' failed before first chunk (%s); "
                    "failing over to next provider",
                    routed_provider.name,
                    type(exc).__name__,
                )
                await self._record_provider_failure(
                    routing_result, routed_provider, exc
                )
                excluded.add(routed_provider.name)
                continue

        # Every provider failed before producing a chunk.
        logger.error("Legacy streaming failed on all providers: %s", failures)
        yield f"data: {json.dumps({'error': 'All providers failed', 'done': True})}\n\n"

    async def _build_context(
        self,
        request: ChatCompletionRequest,
        correlation_id: str | None,
    ) -> ChatContext:
        return await build_generate_context(
            request=request,
            gateway_service=self._gateway_service,
            correlation_id=correlation_id,
        )

    @staticmethod
    def _build_provider_request_data(
        request: ChatCompletionRequest,
        messages: list[dict[str, str]],
        selected_model: str,
    ) -> dict[str, Any]:
        return {
            "messages": messages,
            "model": selected_model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": bool(request.stream),
        }

    @staticmethod
    def _format_openai_chunk(
        request_id: str,
        model: str,
        chunk: ProviderChunk,
    ) -> dict[str, Any]:
        return {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": chunk.role,
                        "content": chunk.content,
                    },
                    "finish_reason": chunk.finish_reason,
                }
            ],
        }

    @staticmethod
    def _format_openai_error_chunk(
        request_id: str,
        model: str,
        exc: Exception,
        correlation_id: str | None,
    ) -> str:
        problem = map_exception_to_problem(exc, correlation_id)
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "",
                    },
                    "finish_reason": "error",
                }
            ],
            "error": problem.model_dump(),
        }
        return f"data: {json.dumps(error_chunk)}\n\n"


def translate_legacy_get(task_id: str, goblin: str, task: str) -> ChatCompletionRequest:
    """Translate GET /v1/stream query args into canonical chat request shape."""

    _ = task_id  # Retained for backward-compatible signature.
    system_content = f"You are a helpful AI assistant called {goblin or 'Goblin'}."
    return ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=f"Execute task: {task}"),
        ],
        stream=True,
    )


def translate_legacy_post(body: StreamChatRequest) -> ChatCompletionRequest:
    """Translate POST /v1/stream body into canonical chat request shape."""

    return ChatCompletionRequest(
        messages=[ChatMessage(role=m.role, content=m.content) for m in body.messages],
        model=body.model,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        stream=True,
    )


__all__ = [
    "ChatService",
    "StreamChatMessage",
    "StreamChatRequest",
    "translate_legacy_get",
    "translate_legacy_post",
]
