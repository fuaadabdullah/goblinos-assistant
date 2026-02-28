from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import Request
from pydantic import BaseModel

from ..errors import map_exception_to_problem
from .chat_context import ChatContext, build_generate_context
from .imports import get_gateway_service
from .provider_chunk import ProviderChunk
from .request_validation import ChatCompletionRequest, ChatMessage
from .response_builder import ChatCompletionResponse
from .routing_provider_selector import RoutingProviderSelector
from .token_accounting import TokenAccountingService

logger = logging.getLogger(__name__)


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

        routed_provider, routing_result = await self._selector.select_for_request(
            request=request,
            http_request=http_request,
            messages=context.messages,
            gateway_result=context.gateway_result,
        )

        request_data = self._build_provider_request_data(
            request=request,
            messages=context.messages,
            selected_model=routed_provider.model,
        )
        completion = await routed_provider.chat_completion(request_data)

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

    async def stream(
        self,
        request: ChatCompletionRequest,
        http_request: Request,
    ) -> AsyncGenerator[str, None]:
        correlation_id = getattr(http_request.state, "correlation_id", None)
        context = await self._build_context(request, correlation_id)

        routed_provider, _routing_result = await self._selector.select_for_request(
            request=request,
            http_request=http_request,
            messages=context.messages,
            gateway_result=context.gateway_result,
        )

        request_data = self._build_provider_request_data(
            request=request,
            messages=context.messages,
            selected_model=routed_provider.model,
        )

        chunk_id = routed_provider.request_id or correlation_id or str(uuid.uuid4())
        try:
            async for chunk in routed_provider.stream_chat_completion(request_data):
                payload = self._format_openai_chunk(
                    request_id=chunk_id,
                    model=routed_provider.model,
                    chunk=chunk,
                )
                yield f"data: {json.dumps(payload)}\\n\\n"
        except Exception as exc:  # noqa: BLE001
            problem = map_exception_to_problem(exc, correlation_id)
            error_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": routed_provider.model,
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
            yield f"data: {json.dumps(error_chunk)}\\n\\n"

        yield "data: [DONE]\\n\\n"

    async def stream_legacy(
        self,
        request: ChatCompletionRequest,
        http_request: Request,
    ) -> AsyncGenerator[str, None]:
        correlation_id = getattr(http_request.state, "correlation_id", None)
        context = await self._build_context(request, correlation_id)

        routed_provider, _routing_result = await self._selector.select_for_request(
            request=request,
            http_request=http_request,
            messages=context.messages,
            gateway_result=context.gateway_result,
        )

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
            "\\n\\n"
        )

        try:
            async for chunk in routed_provider.stream_chat_completion(request_data):
                if chunk.content:
                    tok_count = self._token_accountant.count_tokens(chunk.content)
                    total_tokens += tok_count
                    yield (
                        "data: "
                        f"{json.dumps({'content': chunk.content, 'token_count': tok_count, 'done': False})}"
                        "\\n\\n"
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
            yield f"data: {json.dumps(completion_payload)}\\n\\n"

        except Exception as exc:  # noqa: BLE001
            logger.error("Legacy streaming failed: %s", exc)
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\\n\\n"

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
