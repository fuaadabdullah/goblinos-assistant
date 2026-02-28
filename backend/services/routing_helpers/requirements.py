"""
Requirement checking and validation for provider routing.

Handles model, context window, vision capability, and other requirement checks.
"""

import logging
from typing import Dict, Any, Optional

from ..local_llm_routing import Intent, LatencyTarget

logger = logging.getLogger(__name__)


def check_model_requirement(
    provider: Dict[str, Any], requirements: Dict[str, Any]
) -> bool:
    """
    Check if provider meets model requirements.

    Args:
        provider: Provider information
        requirements: Requirements dict

    Returns:
        True if model requirement is met or not specified
    """
    if "model" not in requirements:
        return True

    required_model = requirements["model"]
    return any(model["id"] == required_model for model in provider["models"])


def check_context_window_requirement(
    provider: Dict[str, Any], requirements: Dict[str, Any]
) -> bool:
    """
    Check if provider meets context window requirements.

    Args:
        provider: Provider information
        requirements: Requirements dict

    Returns:
        True if context window requirement is met or not specified
    """
    if "min_context_window" not in requirements:
        return True

    min_window = requirements["min_context_window"]
    return any(model["context_window"] >= min_window for model in provider["models"])


def check_vision_capability_requirement(
    provider: Dict[str, Any], requirements: Dict[str, Any]
) -> bool:
    """
    Check if provider meets vision capability requirements.

    Args:
        provider: Provider information
        requirements: Requirements dict

    Returns:
        True if vision requirement is met or not required
    """
    vision_required = requirements.get("vision_required", False)
    if not vision_required:
        return True

    return "vision" in provider["capabilities"]


def check_provider_requirements(
    provider: Dict[str, Any], requirements: Dict[str, Any]
) -> bool:
    """
    Check if provider meets all additional requirements.

    Args:
        provider: Provider information
        requirements: Requirements to check

    Returns:
        True if all requirements are met
    """
    # Check model requirements
    if not check_model_requirement(provider, requirements):
        return False

    # Check context window requirements
    if not check_context_window_requirement(provider, requirements):
        return False

    # Check vision capability
    if not check_vision_capability_requirement(provider, requirements):
        return False

    return True


def extract_local_llm_routing_parameters(
    requirements: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extract and validate parameters for local LLM routing.

    Args:
        requirements: Request requirements

    Returns:
        Dict with extracted parameters or None if invalid
    """
    # Extract routing parameters
    messages = requirements.get("messages", [])
    if not messages:
        return None

    # Get optional routing hints
    intent = requirements.get("intent")
    if intent and isinstance(intent, str):
        try:
            intent = Intent(intent)
        except ValueError:
            intent = None

    latency_target = requirements.get("latency_target", "medium")
    if isinstance(latency_target, str):
        try:
            latency_target = LatencyTarget(latency_target)
        except ValueError:
            latency_target = LatencyTarget.MEDIUM

    context_provided = requirements.get("context")
    cost_priority = requirements.get("cost_priority", False)

    # Check autoscaling service for rate limiting and fallback
    client_ip = requirements.get("client_ip", "unknown")
    user_id = requirements.get("user_id")

    return {
        "messages": messages,
        "intent": intent,
        "latency_target": latency_target,
        "context_provided": context_provided,
        "cost_priority": cost_priority,
        "client_ip": client_ip,
        "user_id": user_id,
    }
