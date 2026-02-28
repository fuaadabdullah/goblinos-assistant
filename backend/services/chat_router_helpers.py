from __future__ import annotations

from typing import Any

from .chat_context import build_generate_context
from .types import GatewayCheckResult


async def check_gateway_and_prepare(
    request: Any,
    gateway_service: Any,
) -> tuple[list[dict], GatewayCheckResult]:
    """Validate request, prepare messages, and run gateway checks."""

    context = await build_generate_context(request, gateway_service)
    return context.messages, context.gateway_result


__all__ = ["check_gateway_and_prepare"]
