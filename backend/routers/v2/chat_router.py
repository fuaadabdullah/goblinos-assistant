from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from ...auth.dependencies import require_scope
from ...auth.policies import AuthScope
from ...chat_router import get_chat_service
from ...schemas.v2.chat import ChatCompletionRequest
from ...services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat/completions", response_model=None)
async def create_chat_completion_v2(
    request: ChatCompletionRequest,
    req: Request,
    response: Response,
    chat_service: ChatService = Depends(get_chat_service),
    _scopes: List[str] = Depends(require_scope(AuthScope.WRITE_CONVERSATIONS)),
):
    """Version 2 chat completion scaffold delegating to shared ChatService."""

    response.headers["X-API-Version"] = "v2"
    if request.stream:
        return StreamingResponse(
            chat_service.stream(request, req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-API-Version": "v2",
            },
        )

    return await chat_service.complete(request, req)


__all__ = ["router"]

