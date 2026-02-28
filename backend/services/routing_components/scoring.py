"""Provider scoring helpers for routing."""

from typing import Any

from .. import routing_helpers


def calculate_cost_penalty(provider: dict[str, Any], capability: str) -> float:
    """Calculate cost penalty for provider."""
    return routing_helpers.calculate_cost_penalty(provider, capability)


def calculate_cost_penalty_with_budget(
    provider: dict[str, Any],
    capability: str,
    cost_budget: float | None = None,
) -> float:
    """Calculate cost penalty considering budget constraints."""
    return routing_helpers.calculate_cost_penalty_with_budget(provider, capability, cost_budget)


def calculate_capability_bonus(
    provider: dict[str, Any],
    capability: str,
    requirements: dict[str, Any] | None = None,
) -> float:
    """Calculate bonus for capability match."""
    return routing_helpers.calculate_capability_bonus(provider, capability, requirements)


def get_latency_weight(latency_priority: str) -> float:
    """Get latency weight multiplier based on priority."""
    weights = {
        "ultra_low": 2.0,
        "low": 1.5,
        "medium": 1.0,
        "high": 0.7,
    }
    return weights.get(latency_priority, 1.0)


async def calculate_provider_health_score(db, provider_id: int) -> float:
    """Get health score for provider based on recent metrics."""
    return await routing_helpers.calculate_provider_health_score(db, provider_id)


async def calculate_provider_performance_bonus(db, provider_id: int) -> float:
    """Get performance bonus based on recent metrics."""
    return await routing_helpers.calculate_provider_performance_bonus(db, provider_id)


async def calculate_provider_sla_score(
    latency_monitor,
    db,
    provider: dict[str, Any],
    sla_target_ms: float,
) -> float:
    """Calculate SLA compliance score for a provider."""
    return await routing_helpers.calculate_provider_sla_score(
        latency_monitor, db, provider, sla_target_ms
    )


async def calculate_provider_score(
    provider: dict[str, Any],
    capability: str,
    requirements: dict[str, Any] | None,
    sla_target_ms: float | None,
    cost_budget: float | None,
    latency_priority: str | None,
    db,
    latency_monitor,
    cost_budget_weights: dict[str, float],
) -> float:
    """Calculate a score for a provider based on multiple factors including SLA and cost."""
    base_score = 50.0

    health_score = await calculate_provider_health_score(db, provider["id"])
    base_score += health_score * cost_budget_weights["latency_priority"]

    priority_bonus = provider["priority"] * 2.0
    base_score += priority_bonus

    if sla_target_ms:
        sla_score = await calculate_provider_sla_score(latency_monitor, db, provider, sla_target_ms)
        base_score += sla_score * cost_budget_weights["sla_compliance"]

    cost_penalty = calculate_cost_penalty_with_budget(provider, capability, cost_budget)
    base_score -= cost_penalty * cost_budget_weights["cost_priority"]

    performance_bonus = await calculate_provider_performance_bonus(db, provider["id"])
    latency_weight = 1.0
    if latency_priority:
        latency_weight = get_latency_weight(latency_priority)
    base_score += performance_bonus * latency_weight

    capability_bonus = calculate_capability_bonus(provider, capability, requirements)
    base_score += capability_bonus

    return max(0.0, min(100.0, base_score))
