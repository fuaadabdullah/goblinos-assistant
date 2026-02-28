"""
Helper functions for RoutingService to reduce cognitive complexity.

This module contains utility functions extracted from RoutingService
to improve maintainability and reduce method complexity.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional

from .autoscaling_service import FallbackLevel
from .local_llm_routing import (
    select_model,
    get_system_prompt,
    get_routing_explanation,
    detect_intent,
    get_context_length,
    Intent,
    LatencyTarget,
)

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
        autoscaling_service: Autoscaling service instance
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
        # This would need to be passed back to the main method
        return {
            "emergency_routing": True,
            "capability": capability,
            "requirements": requirements,
            "request_id": request_id,
        }

    # If cheap model fallback requested, modify requirements
    if fallback_level == FallbackLevel.CHEAP_MODEL:
        requirements = requirements or {}
        # Support both routing-service callers (RoutingService has autoscaling_service)
        # and legacy tests that pass the autoscaling service directly.
        cheap_model = getattr(routing_service, "cheap_fallback_model", None)
        if not cheap_model and getattr(routing_service, "autoscaling_service", None) is not None:
            cheap_model = getattr(routing_service.autoscaling_service, "cheap_fallback_model", None)
        requirements["model"] = cheap_model or "goblin-simple-llama-1b"
        requirements["fallback_mode"] = True
        logger.info(f"Using cheap fallback model for request {request_id}")

    return {
        "continue_normal_routing": True,
        "requirements": requirements,
        "fallback_level": fallback_level,
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
            logger.warning("Rate limit check failed; continuing without rate limiting", extra={"error": str(e)})
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
            from .local_llm_routing import estimate_token_count

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

    # Ensure the routing result includes a concrete model choice so callers can execute.
    # Local LLM routing already sets provider["model"]; cloud routing must pick one.
    def _coerce_model_ids(models_value: Any) -> list[str]:
        if not models_value:
            return []
        if isinstance(models_value, list):
            out: list[str] = []
            for item in models_value:
                if isinstance(item, str) and item:
                    out.append(item)
                elif isinstance(item, dict):
                    mid = item.get("id") or item.get("name") or item.get("model")
                    if isinstance(mid, str) and mid:
                        out.append(mid)
            return out
        return []

    def _pick_default_model(provider: Dict[str, Any], reqs: Dict[str, Any]) -> Optional[str]:
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
            "siliconeflow": os.getenv("SILICONEFLOW_DEFAULT_MODEL") or os.getenv("SILICONFLOW_DEFAULT_MODEL"),
            "deepseek": os.getenv("DEEPSEEK_DEFAULT_MODEL"),
        }
        preferred = (env_defaults.get(provider_name) or "").strip()
        if preferred and preferred in model_ids:
            return preferred

        return model_ids[0]

    chosen_model = selected_provider.get("model")
    if not (isinstance(chosen_model, str) and chosen_model.strip()):
        chosen_model = _pick_default_model(selected_provider, requirements or {})
        if chosen_model:
            selected_provider["model"] = chosen_model

    if not (isinstance(selected_provider.get("model"), str) and selected_provider["model"].strip()):
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

    # Best-effort: provide a system prompt even for cloud providers to keep behavior consistent.
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
        system_prompt = get_system_prompt(detected)
    except Exception:
        system_prompt = None

    return {
        "success": True,
        "request_id": request_id,
        "provider": selected_provider,
        "capability": capability,
        "requirements": requirements or {},
        "is_fallback": False,
        "system_prompt": system_prompt,
    }


async def fetch_recent_provider_metrics(db_session, provider_id: int, hours: int = 1):
    """
    Fetch recent metrics for a provider from the database.

    Args:
        db_session: Database session
        provider_id: Provider ID to fetch metrics for
        hours: Number of hours to look back

    Returns:
        List of ProviderMetric objects
    """
    from datetime import datetime, timedelta
    import asyncio

    one_hour_ago = datetime.utcnow() - timedelta(hours=hours)

    def _sync_get_metrics():
        from models import ProviderMetric

        return (
            db_session.query(ProviderMetric)
            .filter(
                ProviderMetric.provider_id == provider_id,
                ProviderMetric.timestamp >= one_hour_ago,
            )
            .all()
        )

    return await asyncio.to_thread(_sync_get_metrics)


def calculate_health_rate(metrics) -> float:
    """
    Calculate the health rate from provider metrics.

    Args:
        metrics: List of ProviderMetric objects

    Returns:
        Health rate (0.0 to 1.0)
    """
    if not metrics:
        return 0.0

    total_metrics = len(metrics)
    healthy_count = sum(1 for m in metrics if m.is_healthy)
    return healthy_count / total_metrics if total_metrics > 0 else 0


def calculate_average_response_time(metrics) -> float:
    """
    Calculate average response time from provider metrics.

    Args:
        metrics: List of ProviderMetric objects

    Returns:
        Average response time in milliseconds
    """
    response_times = [m.response_time_ms for m in metrics if m.response_time_ms]
    if not response_times:
        return 1000.0  # Default fallback

    return sum(response_times) / len(response_times)


def calculate_response_time_score(avg_response_time: float) -> float:
    """
    Calculate response time score based on average response time.

    Args:
        avg_response_time: Average response time in milliseconds

    Returns:
        Response time score (0-25, higher is better)
    """
    # Response time score (faster = better, max 2000ms = 0 points)
    return max(0, 25 - (avg_response_time / 80))


def calculate_overall_health_score(
    health_rate: float, response_time_score: float
) -> float:
    """
    Calculate overall health score combining health rate and response time.

    Args:
        health_rate: Health rate (0.0 to 1.0)
        response_time_score: Response time score

    Returns:
        Overall health score (-50 to 75)
    """
    # Health score: -50 (all unhealthy) to 50 (all healthy)
    health_score = (health_rate - 0.5) * 100
    return health_score + response_time_score


async def calculate_provider_health_score(db_session, provider_id: int) -> float:
    """
    Calculate comprehensive health score for a provider.

    Args:
        db_session: Database session
        provider_id: Provider ID

    Returns:
        Health score (-50 to 75)
    """
    # Fetch recent metrics
    metrics = await fetch_recent_provider_metrics(db_session, provider_id)

    if not metrics:
        return 0.0  # No data = neutral

    # Calculate health rate
    health_rate = calculate_health_rate(metrics)

    # Calculate average response time
    avg_response_time = calculate_average_response_time(metrics)

    # Calculate response time score
    response_time_score = calculate_response_time_score(avg_response_time)

    # Calculate overall health score
    return calculate_overall_health_score(health_rate, response_time_score)


def should_use_fallback_based_on_conditions(
    sla_target_ms: Optional[float] = None,
    latency_priority: Optional[str] = None,
) -> bool:
    """
    Check if fallback conditions are met based on SLA and latency priority.

    Args:
        sla_target_ms: SLA target response time
        latency_priority: Latency priority level

    Returns:
        True if fallback conditions are met
    """
    # If no SLA target or low latency priority, don't fallback
    if not sla_target_ms or latency_priority in ["medium", "high"]:
        return False
    return True


async def check_provider_sla_compliance(
    latency_monitor,
    provider: Dict[str, Any],
    sla_target: float,
) -> bool:
    """
    Check if a provider meets SLA compliance.

    Args:
        latency_monitor: Latency monitoring service
        provider: Provider information
        sla_target: SLA target in milliseconds

    Returns:
        True if provider is SLA compliant
    """
    if not provider.get("models"):
        return False

    model_name = provider["models"][0]["id"]
    try:
        sla_check = await latency_monitor.check_sla_compliance(
            provider["name"], model_name, sla_target
        )
        return sla_check.get("compliant", False)
    except Exception as e:
        logger.warning(f"Failed to check SLA for provider {provider['name']}: {e}")
        return False


async def calculate_sla_compliance_rate(
    latency_monitor,
    providers: list,
    sla_target: float,
) -> float:
    """
    Calculate the SLA compliance rate for top providers.

    Args:
        latency_monitor: Latency monitoring service
        providers: List of providers to check
        sla_target: SLA target in milliseconds

    Returns:
        Compliance rate (0.0 to 1.0)
    """
    compliant_providers = 0
    total_checked = 0

    # Check top 3 providers
    for provider in providers[:3]:
        is_compliant = await check_provider_sla_compliance(
            latency_monitor, provider, sla_target
        )
        total_checked += 1
        if is_compliant:
            compliant_providers += 1

    return compliant_providers / total_checked if total_checked > 0 else 0


def should_use_fallback_based_on_compliance(
    compliance_rate: float,
    sla_target: float,
) -> bool:
    """
    Determine if fallback should be used based on SLA compliance rate.

    Args:
        compliance_rate: SLA compliance rate (0.0 to 1.0)
        sla_target: SLA target in milliseconds

    Returns:
        True if fallback should be used
    """
    # If less than 50% of providers meet SLA, use fallback
    if compliance_rate < 0.5:
        logger.info(
            f"Using fallback: only {compliance_rate:.1%} providers meet SLA target of {sla_target}ms"
        )
        return True
    return False


async def should_use_latency_fallback(
    latency_monitor,
    providers: list,
    sla_target_ms: Optional[float] = None,
    latency_priority: Optional[str] = None,
    default_sla_targets: Optional[Dict[str, int]] = None,
) -> bool:
    """
    Check if we should use fallback to local models due to latency issues.

    Args:
        latency_monitor: Latency monitoring service
        providers: Available providers
        sla_target_ms: SLA target response time
        latency_priority: Latency priority level
        default_sla_targets: Default SLA targets by priority

    Returns:
        True if fallback should be used
    """
    # Check if fallback conditions are met
    if not should_use_fallback_based_on_conditions(sla_target_ms, latency_priority):
        return False

    # Determine SLA target
    sla_target = sla_target_ms or (default_sla_targets or {}).get(
        latency_priority, 2000
    )

    # Calculate SLA compliance rate
    compliance_rate = await calculate_sla_compliance_rate(
        latency_monitor, providers, sla_target
    )

    # Determine if fallback should be used
    return should_use_fallback_based_on_compliance(compliance_rate, sla_target)


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

    # Try to find Ollama provider
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


async def fetch_recent_provider_metrics_limited(
    db_session, provider_id: int, limit: int = 10, hours: int = 1
):
    """
    Fetch recent metrics for a provider from the database with a limit.

    Args:
        db_session: Database session
        provider_id: Provider ID to fetch metrics for
        limit: Maximum number of metrics to fetch
        hours: Number of hours to look back

    Returns:
        List of ProviderMetric objects
    """
    from datetime import datetime, timedelta
    import asyncio

    one_hour_ago = datetime.utcnow() - timedelta(hours=hours)

    def _sync_get_recent_metrics():
        from models import ProviderMetric

        return (
            db_session.query(ProviderMetric)
            .filter(
                ProviderMetric.provider_id == provider_id,
                ProviderMetric.timestamp >= one_hour_ago,
            )
            .order_by(ProviderMetric.timestamp.desc())
            .limit(limit)
            .all()
        )

    return await asyncio.to_thread(_sync_get_recent_metrics)


def calculate_performance_bonus_from_response_time(avg_response_time: float) -> float:
    """
    Calculate performance bonus based on average response time.

    Args:
        avg_response_time: Average response time in milliseconds

    Returns:
        Performance bonus (0-15)
    """
    # Bonus for faster response times (under 500ms = 15 points, over 2000ms = 0)
    if avg_response_time <= 500:
        return 15.0
    elif avg_response_time >= 2000:
        return 0.0
    else:
        return 15.0 * (2000 - avg_response_time) / 1500


async def calculate_provider_performance_bonus(db_session, provider_id: int) -> float:
    """
    Calculate performance bonus for a provider based on recent metrics.

    Args:
        db_session: Database session
        provider_id: Provider ID

    Returns:
        Performance bonus (0-15)
    """
    # Get recent metrics
    metrics = await fetch_recent_provider_metrics_limited(db_session, provider_id)

    if not metrics:
        return 0.0

    # Average response time
    response_times = [m.response_time_ms for m in metrics if m.response_time_ms]
    if not response_times:
        return 0.0

    avg_response_time = sum(response_times) / len(response_times)

    # Calculate performance bonus
    return calculate_performance_bonus_from_response_time(avg_response_time)


async def check_sla_compliance_with_monitor(
    latency_monitor,
    provider: Dict[str, Any],
    sla_target_ms: float,
) -> Dict[str, Any]:
    """
    Check SLA compliance for a provider using latency monitoring.

    Args:
        latency_monitor: Latency monitoring service
        provider: Provider information
        sla_target_ms: SLA target in milliseconds

    Returns:
        Dict with SLA compliance information
    """
    if not provider.get("models"):
        return {"data_available": False}

    model_name = provider["models"][0]["id"]
    try:
        sla_check = await latency_monitor.check_sla_compliance(
            provider["name"], model_name, sla_target_ms
        )
        return sla_check
    except Exception as e:
        logger.warning(f"Failed to check SLA compliance for {provider['name']}: {e}")
        return {"data_available": False}


def calculate_sla_score_from_compliance(
    sla_check: Dict[str, Any],
    sla_target_ms: float,
) -> float:
    """
    Calculate SLA score based on compliance check results.

    Args:
        sla_check: SLA compliance check result
        sla_target_ms: SLA target in milliseconds

    Returns:
        SLA score (-20 to 20)
    """
    if not sla_check.get("data_available"):
        return 0.0

    if sla_check.get("compliant"):
        return 20.0  # Full bonus for SLA compliance
    else:
        # Partial penalty based on how far off SLA we are
        current_p95 = sla_check.get("current_p95", float("inf"))
        if current_p95 > sla_target_ms:
            overrun_ratio = current_p95 / sla_target_ms
            return max(-20.0, 10.0 - (overrun_ratio - 1) * 15.0)

    return 0.0


async def calculate_provider_sla_score(
    latency_monitor,
    db_session,
    provider: Dict[str, Any],
    sla_target_ms: float,
) -> float:
    """
    Calculate comprehensive SLA score for a provider.

    Args:
        latency_monitor: Latency monitoring service
        db_session: Database session
        provider: Provider information
        sla_target_ms: SLA target in milliseconds

    Returns:
        SLA score (-20 to 20)
    """
    # Try to get SLA compliance from latency monitoring
    sla_check = await check_sla_compliance_with_monitor(
        latency_monitor, provider, sla_target_ms
    )

    sla_score = calculate_sla_score_from_compliance(sla_check, sla_target_ms)

    if sla_check.get("data_available"):
        return sla_score

    # Fallback to basic health check
    try:
        health_score = await calculate_provider_health_score(db_session, provider["id"])
        # Convert health score to SLA score (rough approximation)
        return health_score * 0.4
    except Exception:
        return 0.0


def calculate_capability_bonus(
    provider: Dict[str, Any],
    capability: str,
    requirements: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Calculate bonus for capability match.

    Args:
        provider: Provider info
        capability: Required capability
        requirements: Additional requirements

    Returns:
        Capability bonus (0-10)
    """
    bonus = 0.0

    # Base capability match
    if capability in provider["capabilities"]:
        bonus += 5.0

    # Specific model requirement
    if requirements and "model" in requirements:
        required_model = requirements["model"]
        if any(model["id"] == required_model for model in provider["models"]):
            bonus += 5.0

    return bonus


def calculate_cost_penalty(provider: Dict[str, Any], capability: str) -> float:
    """
    Calculate cost penalty for provider.

    Args:
        provider: Provider info
        capability: Required capability

    Returns:
        Cost penalty (0-20, higher = more expensive)
    """
    # Find cheapest model for capability
    min_cost = float("inf")
    for model in provider["models"]:
        if capability in model["capabilities"]:
            # Use input token cost as proxy
            cost = model["pricing"].get("input", 0.002)
            min_cost = min(min_cost, cost)

    if min_cost == float("inf"):
        return 10.0  # Default penalty

    # Penalty based on cost relative to baseline (0.001 = 0 penalty, 0.01 = 20 penalty)
    return min(20.0, (min_cost - 0.001) * 2000)


def calculate_cost_penalty_with_budget(
    provider: Dict[str, Any],
    capability: str,
    cost_budget: Optional[float] = None,
) -> float:
    """
    Calculate cost penalty considering budget constraints.

    Args:
        provider: Provider info
        capability: Required capability
        cost_budget: Maximum cost per request in USD

    Returns:
        Cost penalty (0-30, higher = more expensive or over budget)
    """
    # Get base cost penalty
    base_penalty = calculate_cost_penalty(provider, capability)

    # If no budget specified, return base penalty
    if cost_budget is None:
        return base_penalty

    # Find minimum cost for capability
    min_cost = float("inf")
    for model in provider["models"]:
        if capability in model["capabilities"]:
            # Use input token cost as proxy (rough estimate)
            cost = model["pricing"].get("input", 0.002)
            min_cost = min(min_cost, cost)

    if min_cost == float("inf"):
        return base_penalty + 10.0  # Penalty for no pricing info

    # Check if within budget
    if min_cost <= cost_budget:
        # Within budget, reduce penalty
        return base_penalty * 0.5
    else:
        # Over budget, significant penalty
        budget_overrun = min_cost / cost_budget
        return base_penalty + min(20.0, (budget_overrun - 1) * 10.0)


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


async def find_ollama_provider(db_session) -> Optional[Any]:
    """
    Find the active Ollama provider from the database.

    Args:
        db_session: Database session

    Returns:
        Provider object or None
    """

    def _sync_find_provider():
        from models import Provider  # Import here to avoid circular imports

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
