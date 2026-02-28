"""
Provider selection and fallback logic for routing.

Handles provider candidate selection, scoring, ranking, and fallback decisions.
"""

import logging
import os
from typing import Dict, Any, Optional, List

from ..local_llm_routing import detect_intent, get_system_prompt, Intent

logger = logging.getLogger(__name__)


def _coerce_model_ids(models_value: Any) -> List[str]:
    """
    Coerce models value to list of model IDs.

    Args:
        models_value: Value from provider models field

    Returns:
        List of model ID strings
    """
    if not models_value:
        return []
    if isinstance(models_value, list):
        out: List[str] = []
        for item in models_value:
            if isinstance(item, str) and item:
                out.append(item)
            elif isinstance(item, dict):
                mid = item.get("id") or item.get("name") or item.get("model")
                if isinstance(mid, str) and mid:
                    out.append(mid)
        return out
    return []


def _pick_default_model(
    provider: Dict[str, Any], reqs: Dict[str, Any]
) -> Optional[str]:
    """
    Pick a default model from provider based on requirements.

    Args:
        provider: Provider information
        reqs: Requirements dict

    Returns:
        Model ID or None
    """
    explicit = reqs.get("model")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    model_ids = _coerce_model_ids(provider.get("models"))
    if not model_ids:
        return None

    # Provider-specific defaults (if configured) otherwise first known model.
    provider_name = str(provider.get("name") or "").lower()
    env_defaults = {
        "openai": os.getenv("OPENAI_DEFAULT_MODEL"),
        "openrouter": os.getenv("OPENROUTER_DEFAULT_MODEL"),
        "siliconeflow": os.getenv("SILICONEFLOW_DEFAULT_MODEL")
        or os.getenv("SILICONFLOW_DEFAULT_MODEL"),
        "deepseek": os.getenv("DEEPSEEK_DEFAULT_MODEL"),
    }
    preferred = (env_defaults.get(provider_name) or "").strip()
    if preferred and preferred in model_ids:
        return preferred

    return model_ids[0]


async def handle_provider_selection_and_fallback(
    routing_service,
    capability: str,
    requirements: Dict[str, Any],
    request_id: str,
    sla_target_ms: Optional[float] = None,
    cost_budget: Optional[float] = None,
    latency_priority: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Handle provider selection with fallback logic.

    Args:
        routing_service: Routing service instance
        capability: Required capability
        requirements: Request requirements
        request_id: Request ID for tracking
        sla_target_ms: SLA target response time
        cost_budget: Maximum cost per request
        latency_priority: Latency priority

    Returns:
        Dict with routing result
    """
    # Find suitable providers
    candidates = await routing_service._find_suitable_providers(
        capability, requirements
    )

    if not candidates:
        return {
            "success": False,
            "error": f"No providers available for capability: {capability}",
            "request_id": request_id,
        }

    # Check if we should use fallback due to latency issues
    use_fallback = await routing_service._should_use_fallback(
        candidates, sla_target_ms, latency_priority
    )

    if use_fallback:
        fallback_provider = await routing_service._get_fallback_provider()
        if fallback_provider:
            logger.info(f"Using fallback provider: {fallback_provider['display_name']}")

            # Log the routing decision
            await routing_service._log_routing_request(
                request_id=request_id,
                capability=capability,
                requirements=requirements,
                selected_provider_id=fallback_provider["id"],
                success=True,
            )

            return {
                "success": True,
                "request_id": request_id,
                "provider": fallback_provider,
                "capability": capability,
                "requirements": requirements,
                "is_fallback": True,
                "fallback_reason": "latency_sla_violation",
            }

    # Score and rank providers
    scored_providers = await routing_service._score_providers(
        candidates,
        capability,
        requirements,
        sla_target_ms,
        cost_budget,
        latency_priority,
    )

    if not scored_providers:
        return {
            "success": False,
            "error": "No healthy providers available",
            "request_id": request_id,
        }

    # Select best provider
    selected_provider = scored_providers[0]

    # Ensure the routing result includes a concrete model choice
    chosen_model = selected_provider.get("model")
    if not (isinstance(chosen_model, str) and chosen_model.strip()):
        chosen_model = _pick_default_model(selected_provider, requirements or {})
        if chosen_model:
            selected_provider["model"] = chosen_model

    if not (
        isinstance(selected_provider.get("model"), str)
        and selected_provider["model"].strip()
    ):
        return {
            "success": False,
            "error": "Selected provider did not expose any runnable model (missing models list)",
            "request_id": request_id,
        }

    # Log the routing decision
    await routing_service._log_routing_request(
        request_id=request_id,
        capability=capability,
        requirements=requirements,
        selected_provider_id=selected_provider["id"],
        success=True,
    )

    # Best-effort: provide a system prompt even for cloud providers
    system_prompt = _get_system_prompt_for_provider(requirements)

    return {
        "success": True,
        "request_id": request_id,
        "provider": selected_provider,
        "capability": capability,
        "requirements": requirements or {},
        "is_fallback": False,
        "system_prompt": system_prompt,
    }


def _get_system_prompt_for_provider(
    requirements: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Get system prompt for cloud provider based on requirements.

    Args:
        requirements: Request requirements

    Returns:
        System prompt or None
    """
    try:
        messages = (requirements or {}).get("messages") or []
        intent_hint = (requirements or {}).get("intent")
        detected = None
        if isinstance(intent_hint, Intent):
            detected = intent_hint
        elif isinstance(intent_hint, str):
            try:
                detected = Intent(intent_hint)
            except ValueError:
                detected = None
        if detected is None:
            detected = detect_intent(messages) if messages else Intent.CHAT
        return get_system_prompt(detected)
    except Exception:
        return None


async def handle_routing_error(
    e: Exception,
    request_id: str,
    capability: str,
    requirements: Optional[Dict[str, Any]],
    log_routing_request_func,
) -> Dict[str, Any]:
    """
    Handle routing errors with logging.

    Args:
        e: The exception that occurred
        request_id: Request ID
        capability: Required capability
        requirements: Request requirements
        log_routing_request_func: Function to log routing request

    Returns:
        Error response dict
    """
    logger.error(f"Routing failed for capability {capability}: {e}")

    # Log failed routing
    await log_routing_request_func(
        request_id=request_id,
        capability=capability,
        requirements=requirements,
        success=False,
        error_message=str(e),
    )

    return {"success": False, "error": str(e), "request_id": request_id}
