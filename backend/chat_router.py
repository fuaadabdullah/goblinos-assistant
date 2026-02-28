from __future__ import annotations

import logging
from typing import Annotated, Any, List, TYPE_CHECKING

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import RedirectResponse, StreamingResponse

from .auth.dependencies import require_scope
from .auth.policies import AuthScope
from .schemas.v1.chat import ChatCompletionRequest
from .services import config_processor, request_validation
from .services.chat_router_helpers import check_gateway_and_prepare
from .services.chat_service import (
    ChatService,
    StreamChatRequest,
    translate_legacy_get,
    translate_legacy_post,
)
from .services.imports import get_chat_routing_service

if TYPE_CHECKING:
    from .services.routing import RoutingService

    RoutingServiceType = RoutingService
else:
    RoutingServiceType = Any

logger = logging.getLogger(__name__)

# Re-export get_routing_encryption_key for backward compatibility with tests
get_routing_encryption_key = config_processor.get_routing_encryption_key

# Use explicit path prefixes in route decorators so this router can also host
# the legacy /stream shim under the same module.
router = APIRouter(tags=["chat"])

# Media type constant used for streaming responses
_STREAM_MEDIA_TYPE = "text/event-stream"

def validate_chat_request(request: ChatCompletionRequest) -> None:
    """Backward-compatible wrapper used by legacy callers and tests."""
    return request_validation.validate_chat_request(request)


_check_gateway_and_prepare = check_gateway_and_prepare


def get_chat_service(
    routing_service: "RoutingServiceType" = Depends(get_chat_routing_service),
) -> ChatService:
    return ChatService(routing_service=routing_service)


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    req: Request,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    _scopes: Annotated[List[str], Depends(require_scope(AuthScope.WRITE_CONVERSATIONS))],
):
    """Create a chat completion; supports both sync and SSE transports."""

    if request.stream:
        return StreamingResponse(
            chat_service.stream(request, req),
            media_type=_STREAM_MEDIA_TYPE,
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return await chat_service.complete(request, req)


@router.get("/stream", deprecated=True)
async def legacy_stream_get(
    req: Request,
    task_id: Annotated[str, Query(...)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    _scopes: Annotated[List[str], Depends(require_scope(AuthScope.WRITE_CONVERSATIONS))],
    goblin: Annotated[str, Query()] = "default",
    task: Annotated[str, Query()] = "default task",
):
    """Deprecated shim for legacy GET /v1/stream behavior."""

    logger.info(
        "legacy_stream_get_called",
        extra={
            "correlation_id": getattr(req.state, "correlation_id", "unset") if req else "unset",
            "task_id": task_id,
            "goblin": goblin,
        },
    )

    translated = translate_legacy_get(task_id=task_id, goblin=goblin, task=task)
    return StreamingResponse(
        chat_service.stream_legacy(translated, req),
        media_type=_STREAM_MEDIA_TYPE,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/stream", deprecated=True)
async def legacy_stream_post(
    body: StreamChatRequest,
    req: Request,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    _scopes: Annotated[List[str], Depends(require_scope(AuthScope.WRITE_CONVERSATIONS))],
):
    """Deprecated shim for legacy POST /v1/stream behavior."""

    logger.info(
        "legacy_stream_post_called",
        extra={
            "correlation_id": getattr(req.state, "correlation_id", "unset"),
            "model": body.model,
        },
    )

    translated = translate_legacy_post(body)
    return StreamingResponse(
        chat_service.stream_legacy(translated, req),
        media_type=_STREAM_MEDIA_TYPE,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/essay")
async def essay_redirect():
    """Temporary backward-compatible redirect for /chat/essay -> /essay."""
    return RedirectResponse(url="/essay", status_code=307)
