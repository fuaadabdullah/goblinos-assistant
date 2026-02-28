"""
Chat Controller Helpers.

Utility functions for chat controller operations.
Reduces cognitive complexity by extracting common logic.
"""

from typing import Any, Tuple, List, Dict

from .chat_context import build_generate_context


async def gateway_check_and_prepare(
    request: Any,
    gateway_service: Any,
) -> Tuple[List[Dict], Any]:
    """
    Validate request, prepare messages, and run gateway checks.

    Args:
        request: The chat completion request
        gateway_service: The gateway service instance

    Returns:
        Tuple of (prepared_messages, gateway_result)
    """
    context = await build_generate_context(request, gateway_service)
    return context.messages, context.gateway_result
