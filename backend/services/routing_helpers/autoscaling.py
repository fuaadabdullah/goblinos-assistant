"""
Autoscaling and rate limiting helpers for routing service.

Handles autoscaling checks, rate limiting, and emergency routing logic.
"""

import logging
from typing import Dict, Any, Optional

from ..autoscaling_service import FallbackLevel

logger = logging.getLogger(__name__)


async def handle_autoscaling_and_emergency_routing(
    routing_service,
    capability: str,
    requirements: Optional[Dict[str, Any]],
    request_id: str,
    client_ip: Optional[str] = None,
    user_id: Optional[str] = None,
    request_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Handle autoscaling checks and emergency routing logic.

    Args:
        routing_service: Routing service instance
        capability: Required capability (e.g., "chat", "vision")
        requirements: Additional requirements for the request
        request_id: Request ID for tracking
        client_ip: Client IP for rate limiting
        user_id: User ID for rate limiting
        request_path: Request path for emergency endpoint detection

    Returns:
        Dict with routing result or None if normal routing should continue
    """
    # Check autoscaling and rate limiting first
    autoscaling_result = await routing_service._check_autoscaling(
        client_ip, user_id, request_path, capability
    )

    if not autoscaling_result["allowed"]:
        return {
            "success": False,
            "error": "Rate limit exceeded",
            "fallback_level": autoscaling_result["fallback_level"].value,
            "retry_after": autoscaling_result["retry_after"],
            "request_id": request_id,
        }

    fallback_level = autoscaling_result["fallback_level"]

    # If emergency mode or emergency endpoint, use minimal routing
    if (
        fallback_level == FallbackLevel.EMERGENCY
        or autoscaling_result["emergency_endpoint"]
    ):
        return {
            "emergency_routing": True,
            "capability": capability,
            "requirements": requirements,
            "request_id": request_id,
        }

    # If cheap model fallback requested, modify requirements
    if fallback_level == FallbackLevel.CHEAP_MODEL:
        requirements = requirements or {}
        # Support both routing-service callers and legacy tests
        cheap_model = getattr(routing_service, "cheap_fallback_model", None)
        if not cheap_model and getattr(routing_service, "autoscaling_service", None):
            cheap_model = getattr(
                routing_service.autoscaling_service, "cheap_fallback_model", None
            )
        requirements["model"] = cheap_model or "goblin-simple-llama-1b"
        requirements["fallback_mode"] = True
        logger.info(f"Using cheap fallback model for request {request_id}")

    return {
        "continue_normal_routing": True,
        "requirements": requirements,
        "fallback_level": fallback_level,
    }


async def check_autoscaling_conditions(
    autoscaling_service,
    client_ip: Optional[str],
    user_id: Optional[str],
    request_path: Optional[str],
    capability: str,
) -> Dict[str, Any]:
    """
    Check autoscaling conditions and rate limits.

    Args:
        autoscaling_service: Autoscaling service instance
        client_ip: Client IP for rate limiting
        user_id: User ID for rate limiting
        request_path: Request path for emergency endpoint detection
        capability: Required capability

    Returns:
        Dict with autoscaling check results
    """
    try:
        await autoscaling_service.initialize()

        # Check if emergency mode
        if await autoscaling_service.is_emergency_mode():
            return {
                "allowed": True,
                "fallback_level": FallbackLevel.EMERGENCY,
                "emergency_endpoint": True,
                "retry_after": None,
            }

        # Check if emergency endpoint
        if request_path and await autoscaling_service.is_emergency_endpoint(
            request_path
        ):
            return {
                "allowed": True,
                "fallback_level": FallbackLevel.NORMAL,
                "emergency_endpoint": True,
                "retry_after": None,
            }

        # Check rate limiting
        (
            allowed,
            fallback_level,
            metadata,
        ) = await autoscaling_service.check_rate_limit(client_ip or "unknown", user_id)

        return {
            "allowed": allowed,
            "fallback_level": fallback_level,
            "emergency_endpoint": False,
            "retry_after": metadata.get("cooldown_until"),
        }

    except Exception as e:
        logger.error(f"Autoscaling check failed: {e}")
        # Allow request but log error
        return {
            "allowed": True,
            "fallback_level": FallbackLevel.NORMAL,
            "emergency_endpoint": False,
            "retry_after": None,
        }
