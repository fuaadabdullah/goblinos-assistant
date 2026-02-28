"""
SLA compliance checking for provider routing.

Handles SLA target validation, compliance rate calculations, and fallback decisions.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


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
    providers: List[Dict[str, Any]],
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
    providers: List[Dict[str, Any]],
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
    from .health_metrics import calculate_provider_health_score

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
