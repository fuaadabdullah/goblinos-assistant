"""
Scoring system for provider selection and ranking.

Provides deterministic scoring functions for latency, cost, reliability,
model capability, and other routing factors.
"""

from dataclasses import dataclass
from typing import Dict, Any
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProviderScore:
    """Score for a provider based on multiple factors."""

    provider_id: str
    score: float
    latency_ms: float
    cost_estimate: float
    reliability: float
    model_capability: float
    confidence: float
    metadata: Dict[str, Any]


def score_provider(
    provider_meta: Dict[str, Any], request_context: Dict[str, Any]
) -> ProviderScore:
    """Score a provider for a given request context.

    Args:
        provider_meta: Provider metadata including capabilities, health status, etc.
        request_context: Request context including requirements, constraints, etc.

    Returns:
        ProviderScore with comprehensive scoring information
    """
    provider_id = provider_meta.get("provider_id", "unknown")

    # Extract scoring factors
    latency_score = _score_latency(provider_meta, request_context)
    cost_score = _score_cost(provider_meta, request_context)
    reliability_score = _score_reliability(provider_meta, request_context)
    capability_score = _score_capability(provider_meta, request_context)

    # Apply policy weights
    policy = request_context.get("policy", {})
    latency_weight = policy.get("latency_weight", 0.4)
    cost_weight = policy.get("cost_weight", 0.3)
    reliability_weight = policy.get("reliability_weight", 0.2)
    capability_weight = policy.get("capability_weight", 0.1)

    # Calculate composite score
    composite_score = (
        latency_score * latency_weight
        + cost_score * cost_weight
        + reliability_score * reliability_weight
        + capability_score * capability_weight
    )

    # Calculate confidence based on data freshness and sample size
    confidence = _calculate_confidence(provider_meta)

    return ProviderScore(
        provider_id=provider_id,
        score=composite_score,
        latency_ms=provider_meta.get("avg_latency_ms", 1000),
        cost_estimate=provider_meta.get("cost_estimate", 0.0),
        reliability=provider_meta.get("reliability_score", 0.5),
        model_capability=capability_score,
        confidence=confidence,
        metadata={
            "latency_score": latency_score,
            "cost_score": cost_score,
            "reliability_score": reliability_score,
            "capability_score": capability_score,
            "weights": {
                "latency": latency_weight,
                "cost": cost_weight,
                "reliability": reliability_weight,
                "capability": capability_weight,
            },
        },
    )


def _score_latency(
    provider_meta: Dict[str, Any], request_context: Dict[str, Any]
) -> float:
    """Score provider based on latency performance.

    Returns score from 0.0 (worst) to 1.0 (best).
    """
    avg_latency = provider_meta.get("avg_latency_ms", 5000)
    sla_target = request_context.get("sla_target_ms", 3000)

    if avg_latency <= sla_target:
        return 1.0
    elif avg_latency <= sla_target * 2:
        return 0.7
    elif avg_latency <= sla_target * 5:
        return 0.3
    else:
        return 0.1


def _score_cost(
    provider_meta: Dict[str, Any], request_context: Dict[str, Any]
) -> float:
    """Score provider based on cost efficiency.

    Returns score from 0.0 (worst) to 1.0 (best).
    """
    cost_estimate = provider_meta.get("cost_estimate", 1.0)
    budget = request_context.get("cost_budget", 1.0)

    if cost_estimate <= budget:
        return 1.0
    elif cost_estimate <= budget * 2:
        return 0.5
    else:
        return 0.1


def _score_reliability(
    provider_meta: Dict[str, Any], request_context: Dict[str, Any]
) -> float:
    """Score provider based on reliability metrics.

    Returns score from 0.0 (worst) to 1.0 (best).
    """
    success_rate = provider_meta.get("success_rate", 0.8)
    error_rate = provider_meta.get("error_rate", 0.2)

    # Weighted combination of success and error rates
    return success_rate * 0.8 + (1 - error_rate) * 0.2


def _score_capability(
    provider_meta: Dict[str, Any], request_context: Dict[str, Any]
) -> float:
    """Score provider based on capability match.

    Returns score from 0.0 (worst) to 1.0 (best).
    """
    required_capabilities = request_context.get("required_capabilities", [])
    provider_capabilities = provider_meta.get("capabilities", [])

    if not required_capabilities:
        return 1.0

    matched = 0
    for req_cap in required_capabilities:
        if req_cap in provider_capabilities:
            matched += 1

    return matched / len(required_capabilities) if required_capabilities else 1.0


def _calculate_confidence(provider_meta: Dict[str, Any]) -> float:
    """Calculate confidence in the scoring based on data quality.

    Returns confidence from 0.0 (low) to 1.0 (high).
    """
    sample_size = provider_meta.get("sample_size", 0)
    last_updated = provider_meta.get("last_updated", 0)
    now = time.time()

    # Age factor (newer data = higher confidence)
    age_hours = (now - last_updated) / 3600
    age_factor = max(0.1, 1.0 - (age_hours / 24))  # Degrade over 24 hours

    # Sample size factor
    if sample_size >= 100:
        sample_factor = 1.0
    elif sample_size >= 10:
        sample_factor = 0.7
    elif sample_size >= 1:
        sample_factor = 0.4
    else:
        sample_factor = 0.1

    return age_factor * sample_factor
