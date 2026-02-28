"""
Cost analysis for provider routing.

Handles cost calculations, budget constraints, and cost-based scoring.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


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
