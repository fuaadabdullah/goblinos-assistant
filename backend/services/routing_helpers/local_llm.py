"""
Local LLM routing logic.

Handles local model selection, parameter extraction, and routing for local Ollama providers.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from ..autoscaling_service import FallbackLevel
from ..local_llm_routing import (
    select_model,
    get_system_prompt,
    get_routing_explanation,
    detect_intent,
    get_context_length,
)
from .requirements import extract_local_llm_routing_parameters

logger = logging.getLogger(__name__)


async def find_ollama_provider(db_session) -> Optional[Any]:
    """
    Find the active Ollama provider from the database.

    Args:
        db_session: Database session

    Returns:
        Provider object or None
    """

    def _sync_find_provider():
        from models import Provider

        return (
            db_session.query(Provider)
            .filter(Provider.name == "ollama_gcp", Provider.is_active)
            .first()
        )

    return await asyncio.to_thread(_sync_find_provider)


def build_local_routing_result(
    ollama_provider,
    model_id: str,
    params: Dict[str, Any],
    system_prompt: str,
    explanation: str,
    detected_intent,
    context_length: int,
) -> Dict[str, Any]:
    """
    Build the result dict for local LLM routing.

    Args:
        ollama_provider: Provider object
        model_id: Selected model ID
        params: Model parameters
        system_prompt: System prompt
        explanation: Routing explanation
        detected_intent: Detected intent
        context_length: Context length

    Returns:
        Result dict
    """
    provider_info = {
        "id": ollama_provider.id,
        "name": ollama_provider.name,
        "display_name": ollama_provider.display_name,
        "base_url": ollama_provider.base_url,
        "model": model_id,
        "capabilities": ollama_provider.capabilities,
        "priority": ollama_provider.priority,
    }

    return {
        "provider": provider_info,
        "provider_id": ollama_provider.id,
        "params": params,
        "system_prompt": system_prompt,
        "explanation": explanation,
        "intent": detected_intent.value,
        "context_length": context_length,
    }


async def handle_local_llm_routing(
    autoscaling_service,
    db_session,
    capability: str,
    requirements: Dict[str, Any],
    request_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Handle local LLM routing attempts for chat requests.

    Args:
        autoscaling_service: Autoscaling service instance
        db_session: Database session
        capability: Required capability
        requirements: Request requirements
        request_id: Request ID for tracking

    Returns:
        Dict with local routing result or None if not applicable
    """
    # Only handle chat requests
    if capability != "chat" or not requirements:
        return None

    try:
        # Extract and validate parameters
        params = extract_local_llm_routing_parameters(requirements)
        if not params:
            return None

        # Check rate limiting
        try:
            (
                allowed,
                fallback_level,
                rate_limit_metadata,
            ) = await autoscaling_service.check_rate_limit(
                client_ip=params["client_ip"], user_id=params["user_id"]
            )
        except Exception as e:
            # Redis being down should not break routing in dev/single-instance setups.
            logger.warning(
                "Rate limit check failed; continuing without rate limiting",
                extra={"error": str(e)},
            )
            allowed = True
            fallback_level = FallbackLevel.NORMAL
            rate_limit_metadata = {}

        if not allowed:
            logger.warning(
                f"Request denied due to rate limiting: {rate_limit_metadata}"
            )
            return {
                "error": "Rate limit exceeded",
                "fallback_level": fallback_level.name,
                "retry_after": 60,
                "request_id": request_id,
            }

        # Select model using routing logic
        model_id, model_params = select_model(
            messages=params["messages"],
            intent=params["intent"],
            latency_target=params["latency_target"],
            context_provided=params["context_provided"],
            cost_priority=params["cost_priority"],
            force_cheap_fallback=(fallback_level == FallbackLevel.CHEAP_MODEL),
        )

        # Find Ollama provider
        ollama_provider = await find_ollama_provider(db_session)
        if not ollama_provider:
            logger.warning("Ollama provider not found or not active")
            return None

        # Get system prompt and routing explanation
        detected_intent = params["intent"] or detect_intent(params["messages"])
        system_prompt = get_system_prompt(detected_intent)

        context_length = get_context_length(params["messages"])
        if params["context_provided"]:
            from ..local_llm_routing import estimate_token_count

            context_length += estimate_token_count(params["context_provided"])

        explanation = get_routing_explanation(
            model_id, detected_intent, context_length, params["latency_target"]
        )

        # Build result
        return build_local_routing_result(
            ollama_provider,
            model_id,
            model_params,
            system_prompt,
            explanation,
            detected_intent,
            context_length,
        )

    except Exception as e:
        logger.error(f"Local LLM routing failed: {e}")
        return None


async def process_local_llm_routing_result(
    local_routing: Dict[str, Any],
    request_id: str,
    capability: str,
    requirements: Dict[str, Any],
    log_routing_request_func,
) -> Dict[str, Any]:
    """
    Process the result of local LLM routing.

    Args:
        local_routing: Result from handle_local_llm_routing
        request_id: Request ID
        capability: Required capability
        requirements: Request requirements
        log_routing_request_func: Function to log routing request

    Returns:
        Routing response dict
    """
    # Handle rate limit error
    if "error" in local_routing:
        return {
            "success": False,
            "error": local_routing["error"],
            "fallback_level": local_routing.get("fallback_level"),
            "retry_after": local_routing.get("retry_after", 60),
            "request_id": request_id,
        }

    # Log the routing decision
    await log_routing_request_func(
        request_id=request_id,
        capability=capability,
        requirements=requirements,
        selected_provider_id=local_routing.get("provider_id"),
        success=True,
    )

    return {
        "success": True,
        "request_id": request_id,
        "provider": local_routing["provider"],
        "capability": capability,
        "requirements": requirements,
        "routing_explanation": local_routing.get("explanation"),
        "recommended_params": local_routing.get("params"),
        "system_prompt": local_routing.get("system_prompt"),
    }


async def find_fast_local_model(
    db_session,
    encryption_service,
    adapters: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Find a fast local model for fallback.

    Args:
        db_session: Database session
        encryption_service: Encryption service for API keys
        adapters: Adapter classes

    Returns:
        Provider info for fast local model or None
    """

    def _sync_find_provider():
        from models import Provider

        return (
            db_session.query(Provider)
            .filter(Provider.name == "ollama_gcp", Provider.is_active)
            .first()
        )

    provider = await asyncio.to_thread(_sync_find_provider)

    if not provider:
        return None

    # Check if fast models are available
    try:
        # Get API key - try encrypted first, fall back to plain text
        api_key = None
        if provider.api_key_encrypted:
            try:
                api_key = encryption_service.decrypt(provider.api_key_encrypted)
            except Exception:
                pass
        if not api_key and provider.api_key:
            api_key = provider.api_key
        if not api_key:
            return None

        adapter = adapters["ollama_gcp"](api_key, provider.base_url)
        models = await adapter.list_models()

        # Look for fast models
        fast_models = ["mistral:7b", "llama3.2:3b", "phi3:3.8b", "gemma:2b"]
        for model in models:
            if any(fast_model in model["id"] for fast_model in fast_models):
                return {
                    "id": provider.id,
                    "name": provider.name,
                    "display_name": f"{provider.display_name} (Fallback)",
                    "base_url": provider.base_url,
                    "model": model["id"],
                    "capabilities": provider.capabilities,
                    "priority": provider.priority,
                    "is_fallback": True,
                }
    except Exception as e:
        logger.warning(f"Failed to check fast local models: {e}")

    return None


async def get_fallback_provider_info(
    db_session,
    encryption_service,
    adapters: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Get fallback provider information.

    Args:
        db_session: Database session
        encryption_service: Encryption service for API keys
        adapters: Adapter classes

    Returns:
        Fallback provider info or None
    """
    # Try to find a fast local model first
    fast_model = await find_fast_local_model(db_session, encryption_service, adapters)
    if fast_model:
        return fast_model

    # Fallback to basic model
    def _sync_find_provider():
        from models import Provider

        return (
            db_session.query(Provider)
            .filter(Provider.name == "ollama_gcp", Provider.is_active)
            .first()
        )

    provider = await asyncio.to_thread(_sync_find_provider)
    if provider:
        return {
            "id": provider.id,
            "name": provider.name,
            "display_name": f"{provider.display_name} (Fallback)",
            "base_url": provider.base_url,
            "model": "gemma:2b",  # Default fast fallback
            "capabilities": provider.capabilities,
            "priority": provider.priority,
            "is_fallback": True,
        }

    return None
