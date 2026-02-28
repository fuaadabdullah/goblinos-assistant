"""
Provider scoring module for intelligent LLM provider ranking.

This module scores and ranks LLM providers based on multiple factors:
- Health metrics (error rates, availability)
- Performance metrics (latency, throughput)
- Cost optimization (per-token pricing, budget constraints)
- Capability matching (model features, requirements)
- SLA compliance (response time targets)
- Latency priorities (user-defined performance preferences)

Scoring Algorithm:
    Base score starts at 50 (neutral)
    + Health score (weighted by latency priority): -50 to +75
    + Priority bonus: 0 to +20 (based on provider tier)
    + SLA compliance: -20 to +20 (weighted)
    + Performance bonus: 0 to +15 (weighted by latency priority)
    + Capability bonus: 0 to +10
    - Cost penalty: 0 to -30 (weighted by cost priority)

    Final score: 0 to 100 (clamped)

Dependencies:
    - routing_helpers: Helper functions for metrics calculation
    - routing_config: Configuration and weight parameters
    - latency_monitoring_service: Real-time latency tracking

Author: GoblinOS Team
Last Updated: February 18, 2026
"""

import logging
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from . import routing_helpers
from .routing_config import RoutingConfig
from .latency_monitoring_service import LatencyMonitoringService

logger = logging.getLogger(__name__)


class ProviderScorer:
    """Scores and ranks LLM providers based on multiple factors.

    This class encapsulates all provider scoring logic, making it independently
    testable and reusable across different routing strategies.
    """

    def __init__(
        self,
        config: RoutingConfig,
        db: Optional[Session] = None,
        async_db: Optional[AsyncSession] = None,
        latency_monitor: Optional[LatencyMonitoringService] = None,
    ):
        """Initialize provider scorer.

        Args:
            config: Routing configuration with scoring weights and thresholds
            db: Synchronous database session (for backward compatibility)
            async_db: Async database session (preferred)
            latency_monitor: Latency monitoring service for SLA tracking

        Note:
            Either db or async_db must be provided. If both provided, async_db takes precedence.
        """
        if async_db is not None:
            self.async_db = async_db
            self.db = None
        elif db is not None:
            self.db = db
            self.async_db = None
        else:
            raise ValueError("Either db or async_db must be provided")

        self.config = config
        self.cost_budget_weights = config.cost_budget_weights
        self.latency_monitor = latency_monitor or LatencyMonitoringService()

    async def score_providers(
        self,
        providers: List[Dict[str, Any]],
        capability: str,
        requirements: Optional[Dict[str, Any]] = None,
        sla_target_ms: Optional[float] = None,
        cost_budget: Optional[float] = None,
        latency_priority: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Score and rank providers based on health, performance, cost, and SLA compliance.

        Args:
            providers: List of provider candidates
            capability: Required capability (e.g., 'chat', 'completion', 'embeddings')
            requirements: Additional requirements (model features, context length, etc.)
            sla_target_ms: SLA target response time in milliseconds
            cost_budget: Maximum cost per request in USD
            latency_priority: Latency priority level ('ultra_low', 'low', 'medium', 'high')

        Returns:
            List of providers with scores, sorted by score descending (best first)

        Example:
            >>> scorer = ProviderScorer(config, db=session)
            >>> providers = [{'id': 1, 'name': 'OpenAI', 'priority': 5, ...}]
            >>> scored = await scorer.score_providers(
            ...     providers,
            ...     capability='chat',
            ...     sla_target_ms=1000.0,
            ...     latency_priority='low'
            ... )
            >>> print(scored[0]['score'])  # Best provider score
            85.5
        """
        scored = []

        for provider in providers:
            score = await self._calculate_provider_score(
                provider,
                capability,
                requirements,
                sla_target_ms,
                cost_budget,
                latency_priority,
            )
            if score > 0:  # Only include providers with positive scores (healthy)
                provider_with_score = provider.copy()
                provider_with_score["score"] = score
                scored.append(provider_with_score)

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def _calculate_provider_score(
        self,
        provider: Dict[str, Any],
        capability: str,
        requirements: Optional[Dict[str, Any]] = None,
        sla_target_ms: Optional[float] = None,
        cost_budget: Optional[float] = None,
        latency_priority: Optional[str] = None,
    ) -> float:
        """Calculate a score for a provider based on multiple factors including SLA and cost.

        Scoring breakdown:
            - Base score: 50.0 (neutral starting point)
            - Health score: -50 to +75 (weighted by latency priority)
            - Priority bonus: 0 to +20 (provider tier * 2.0)
            - SLA compliance: -20 to +20 (weighted by sla_compliance weight)
            - Performance bonus: 0 to +15 (weighted by latency priority)
            - Capability bonus: 0 to +10 (exact match bonuses)
            - Cost penalty: 0 to -30 (weighted by cost_priority)

        Args:
            provider: Provider information dict with id, priority, capabilities
            capability: Required capability
            requirements: Additional requirements
            sla_target_ms: SLA target response time in milliseconds
            cost_budget: Maximum cost per request in USD
            latency_priority: Latency priority level

        Returns:
            Score between 0-100 (0 = unusable, 100 = perfect)
        """
        base_score = 50.0  # Start with neutral score

        # Get recent health metrics (error rates, availability)
        health_score = await self._get_health_score(provider["id"])
        base_score += (
            health_score * self.cost_budget_weights["latency_priority"]
        )  # Weighted health score

        # Priority bonus (higher priority providers get preference)
        priority_bonus = provider.get("priority", 0) * 2.0
        base_score += priority_bonus

        # SLA compliance bonus/penalty
        if sla_target_ms:
            sla_score = await self._calculate_sla_score(provider, sla_target_ms)
            base_score += sla_score * self.cost_budget_weights["sla_compliance"]

        # Cost factor with budget consideration
        cost_penalty = self._calculate_cost_penalty_with_budget(
            provider, capability, cost_budget
        )
        base_score -= cost_penalty * self.cost_budget_weights["cost_priority"]

        # Performance bonus (faster = better, adjusted for latency priority)
        performance_bonus = await self._get_performance_bonus(provider["id"])
        latency_weight = 1.0
        if latency_priority:
            latency_weight = self._get_latency_weight(latency_priority)
        base_score += performance_bonus * latency_weight

        # Capability match bonus
        capability_bonus = self._calculate_capability_bonus(
            provider, capability, requirements
        )
        base_score += capability_bonus

        # Ensure score is within bounds
        return max(0.0, min(100.0, base_score))

    async def _get_health_score(self, provider_id: int) -> float:
        """Get health score for provider based on recent metrics.

        Queries recent metrics from the database and calculates health score
        based on error rates, availability, and success rates.

        Args:
            provider_id: Provider ID

        Returns:
            Health score (-50 to 75, higher is better)
            - 75: Perfect health (no errors, high availability)
            - 0: Neutral (average health)
            - -50: Poor health (high error rates, low availability)
        """
        return await routing_helpers.calculate_provider_health_score(
            self.db, provider_id
        )

    def _calculate_cost_penalty(
        self, provider: Dict[str, Any], capability: str
    ) -> float:
        """Calculate cost penalty for provider.

        Args:
            provider: Provider info with pricing data
            capability: Required capability

        Returns:
            Cost penalty (0-20, higher = more expensive)
        """
        return routing_helpers.calculate_cost_penalty(provider, capability)

    async def _get_performance_bonus(self, provider_id: int) -> float:
        """Get performance bonus based on recent metrics.

        Analyzes recent latency and throughput metrics to calculate
        performance bonuses for fast providers.

        Args:
            provider_id: Provider ID

        Returns:
            Performance bonus (0-15, higher is better)
            - 15: Excellent performance (fast response times)
            - 0: Average performance
        """
        return await routing_helpers.calculate_provider_performance_bonus(
            self.db, provider_id
        )

    def _calculate_capability_bonus(
        self,
        provider: Dict[str, Any],
        capability: str,
        requirements: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate bonus for capability match.

        Rewards providers that exactly match the requested capabilities
        and requirements (model features, context length, etc.).

        Args:
            provider: Provider info with capabilities list
            capability: Required capability
            requirements: Additional requirements

        Returns:
            Capability bonus (0-10, higher is better)
            - 10: Perfect match (all requirements met)
            - 5: Partial match (some requirements met)
            - 0: No special advantages
        """
        return routing_helpers.calculate_capability_bonus(
            provider, capability, requirements
        )

    async def _calculate_sla_score(
        self, provider: Dict[str, Any], sla_target_ms: float
    ) -> float:
        """Calculate SLA compliance score for a provider.

        Compares provider's recent latency metrics against the SLA target
        to determine compliance level.

        Args:
            provider: Provider information
            sla_target_ms: SLA target response time in milliseconds

        Returns:
            SLA score (-20 to 20, higher = better SLA compliance)
            - +20: Consistently beats SLA target by 50%+
            - 0: Meets SLA target
            - -20: Frequently misses SLA target
        """
        return await routing_helpers.calculate_provider_sla_score(
            self.latency_monitor, self.db, provider, sla_target_ms
        )

    def _calculate_cost_penalty_with_budget(
        self,
        provider: Dict[str, Any],
        capability: str,
        cost_budget: Optional[float] = None,
    ) -> float:
        """Calculate cost penalty considering budget constraints.

        If cost_budget is provided, applies heavy penalties to providers
        that exceed the budget. Otherwise, applies standard cost penalties
        based on relative pricing.

        Args:
            provider: Provider info with pricing data
            capability: Required capability
            cost_budget: Maximum cost per request in USD (optional)

        Returns:
            Cost penalty (0-30, higher = more expensive or over budget)
            - 30: Significantly over budget or very expensive
            - 15: Moderately expensive
            - 0: Cost-effective or within budget
        """
        return routing_helpers.calculate_cost_penalty_with_budget(
            provider, capability, cost_budget
        )

    def _get_latency_weight(self, latency_priority: str) -> float:
        """Get latency weight multiplier based on priority.

        Adjusts the importance of latency/performance metrics based on
        the user's stated priority level.

        Args:
            latency_priority: Latency priority level
                - 'ultra_low': Critical real-time applications
                - 'low': Interactive applications
                - 'medium': Standard applications (default)
                - 'high': Batch processing, background tasks

        Returns:
            Weight multiplier for latency scoring (0.7 to 2.0)
            - 2.0: Ultra-low latency (2x weight on performance)
            - 1.5: Low latency (50% bonus on performance)
            - 1.0: Medium latency (standard weight)
            - 0.7: High latency tolerance (reduced weight)
        """
        weights = {
            "ultra_low": 2.0,  # Double weight for ultra-low latency
            "low": 1.5,  # 50% bonus for low latency
            "medium": 1.0,  # Normal weight
            "high": 0.7,  # Reduced weight for high latency tolerance
        }
        return weights.get(latency_priority, 1.0)
