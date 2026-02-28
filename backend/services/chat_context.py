from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional

from ..errors import raise_problem
from . import config_processor, request_validation
from .types import GatewayCheckResult

logger = logging.getLogger(__name__)


@dataclass
class ChatContext:
    """Validated chat input plus gateway analysis."""

    messages: list[dict[str, str]]
    gateway_result: GatewayCheckResult
    correlation_id: Optional[str] = None


def _normalize_gateway_result(raw: Any) -> GatewayCheckResult:
    return GatewayCheckResult(
        allowed=getattr(raw, "allowed", True),
        intent=getattr(raw, "intent", None),
        estimated_tokens=getattr(raw, "estimated_tokens", None),
        risk_score=getattr(raw, "risk_score", None),
        fallback_level=getattr(raw, "fallback_level", None),
        retry_after=getattr(raw, "retry_after", None),
        raw=getattr(raw, "__dict__", None),
    )


async def build_generate_context(
    request: request_validation.ChatCompletionRequest,
    gateway_service: Any,
    correlation_id: Optional[str] = None,
) -> ChatContext:
    """Validate request, normalize messages, and run gateway checks."""

    request_validation.validate_chat_request(request)
    messages = config_processor.prepare_messages(request)

    raw_gateway_result = await gateway_service.process_request(
        messages=messages,
        max_tokens=request.max_tokens,
        context=request.context,
    )
    gateway_result = _normalize_gateway_result(raw_gateway_result)

    intent_value = getattr(getattr(gateway_result, "intent", None), "value", gateway_result.intent)
    logger.info(
        "Gateway analysis",
        extra={
            "intent": intent_value,
            "estimated_tokens": gateway_result.estimated_tokens,
            "risk_score": gateway_result.risk_score,
            "allowed": gateway_result.allowed,
        },
    )

    if not gateway_result.allowed:
        raise_problem(
            status=400,
            title="Gateway rejected request",
            detail="Request flagged as high-risk. Reduce token limits or simplify request.",
            type_uri="https://goblin-backend.onrender.com/errors/gateway-denied",
            code="GATEWAY_DENIED",
            instance=correlation_id,
        )

    return ChatContext(
        messages=messages,
        gateway_result=gateway_result,
        correlation_id=correlation_id,
    )


# Alias used by call sites that prefer domain naming over generate naming.
build_chat_context = build_generate_context


__all__ = ["ChatContext", "build_generate_context", "build_chat_context"]
