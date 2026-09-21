from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional

from ..errors import raise_problem
from . import config_processor, request_validation
from . import brave_search, freeserp_search
from .types import GatewayCheckResult

logger = logging.getLogger(__name__)


async def _maybe_add_web_search(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich chat messages with web-search results when warranted.

    Provider selection: Brave Search when BRAVE_API_KEY is set, otherwise
    FreeSerp (keyless, no signup), unless CHAT_WEB_SEARCH_PROVIDER forces
    one of "brave"/"freeserp"/"auto".

    Never breaks chat: any failure (no key, transport error, empty results)
    leaves the messages untouched.
    """
    try:
        name = freeserp_search.provider_name()
        provider = brave_search if name == "brave" else freeserp_search
        query = provider.should_search(messages)
        if not query:
            return messages
        results = await provider.web_search(query)
        if not results:
            return messages
        search_message = {
            "role": "system",
            "content": provider.format_results_for_llm(query, results),
        }
        # Keep it with the other system messages so provider adapters that
        # treat a trailing system message oddly still see it up front.
        insert_at = 0
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "system":
                insert_at = i + 1
            else:
                break
        logger.info(
            "Added %d %s search results for chat query", len(results), name
        )
        return messages[:insert_at] + [search_message] + messages[insert_at:]
    except Exception:
        logger.warning("Chat web-search enrichment failed", exc_info=True)
        return messages


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

    messages = await _maybe_add_web_search(messages)

    return ChatContext(
        messages=messages,
        gateway_result=gateway_result,
        correlation_id=correlation_id,
    )


# Alias used by call sites that prefer domain naming over generate naming.
build_chat_context = build_generate_context


__all__ = ["ChatContext", "build_generate_context", "build_chat_context"]
