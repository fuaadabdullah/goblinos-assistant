"""
Provider Router
Intelligent routing of inference/training requests across GCP, RunPod, and Vast.ai
Integrated with ForgeMonorepo backend configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import structlog

# Use existing backend config
from ..config import settings
from ..config.cloud_providers import get_cloud_config
from .providers.base import BaseProvider, ProviderType, JobType
from .providers.gcp import GCPProvider
from .providers.runpod import RunPodProvider
from .providers.vastai import VastAIProvider

logger = structlog.get_logger()


class RoutingStrategy(str, Enum):
    """Routing strategy for provider selection."""
    COST_OPTIMIZED = "cost_optimized"      # Cheapest available
    LATENCY_OPTIMIZED = "latency_optimized"  # Fastest response
    BALANCED = "balanced"                   # Balance cost and latency
    FAILOVER = "failover"                   # Use backup if primary fails


@dataclass
class RoutingDecision:
    """Result of provider routing decision."""
    provider: ProviderType
    reason: str
    estimated_cost: float
    estimated_latency_ms: float
    fallback_provider: Optional[ProviderType] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderScore:
    """Scoring for provider selection."""
    provider: ProviderType
    cost_score: float  # Lower is better, 0-100
    latency_score: float  # Lower is better, 0-100
    availability_score: float  # Higher is better, 0-100
    total_score: float = 0.0
    
    def calculate_total(
        self, 
        cost_weight: float = 0.4,
        latency_weight: float = 0.3,
        availability_weight: float = 0.3
    ) -> float:
        """Calculate weighted total score (lower is better for cost/latency)."""
        # Invert availability so lower total = better
        self.total_score = (
            self.cost_score * cost_weight +
            self.latency_score * latency_weight +
            (100 - self.availability_score) * availability_weight
        )
        return self.total_score


class ProviderRouter:
    """
    Intelligent router for multi-provider orchestration.
    
    Routes requests based on:
    - Job type (inference, training, batch)
    - Cost constraints
    - Latency requirements
    - Provider availability
    - Budget limits
    """
    
    def __init__(self):
        self.providers: dict[ProviderType, BaseProvider] = {}
        self._cloud_config = get_cloud_config()
        self._initialize_providers()
        self._cost_tracker: dict[str, float] = {}
        self._last_cost_reset = datetime.now(timezone.utc)
        
    def _initialize_providers(self) -> None:
        """Initialize enabled providers using cloud config."""
        # GCP is always available (development)
        self.providers[ProviderType.GCP] = GCPProvider()
        
        if self._cloud_config.runpod.api_key:
            self.providers[ProviderType.RUNPOD] = RunPodProvider()
            
        if self._cloud_config.vastai.api_key:
            self.providers[ProviderType.VASTAI] = VastAIProvider()
            
        logger.info(
            "Providers initialized",
            providers=list(self.providers.keys())
        )
    
    async def route(
        self,
        job_type: JobType,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        required_gpu: Optional[str] = None,
        max_cost_per_hour: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
        prefer_provider: Optional[ProviderType] = None,
    ) -> RoutingDecision:
        """
        Route a job to the optimal provider.
        
        Args:
            job_type: Type of job (inference, training, batch)
            strategy: Routing strategy to use
            required_gpu: Specific GPU type if needed
            max_cost_per_hour: Maximum acceptable cost
            max_latency_ms: Maximum acceptable latency
            prefer_provider: Preferred provider if available
            
        Returns:
            RoutingDecision with selected provider and metadata
        """
        # Check budget
        if not await self._check_budget():
            logger.warning("Budget exceeded, using cheapest fallback")
            return await self._route_budget_exceeded(job_type)
        
        # Get provider scores
        scores = await self._score_providers(
            job_type=job_type,
            required_gpu=required_gpu,
            max_cost_per_hour=max_cost_per_hour,
            max_latency_ms=max_latency_ms,
        )
        
        if not scores:
            raise RuntimeError("No providers available for job")
        
        # Apply strategy
        selected = self._apply_strategy(scores, strategy, prefer_provider)
        
        # Determine fallback
        fallback = self._get_fallback(scores, selected.provider)
        
        return RoutingDecision(
            provider=selected.provider,
            reason=f"Selected via {strategy.value} strategy",
            estimated_cost=selected.cost_score / 10,  # Normalize
            estimated_latency_ms=selected.latency_score * 10,
            fallback_provider=fallback,
            metadata={
                "scores": {s.provider.value: s.total_score for s in scores},
                "strategy": strategy.value,
            }
        )
    
    async def _score_providers(
        self,
        job_type: JobType,
        required_gpu: Optional[str] = None,
        max_cost_per_hour: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
    ) -> list[ProviderScore]:
        """Score all available providers for the job."""
        scores = []
        
        for provider_type, provider in self.providers.items():
            # Check health
            if not await provider.health_check():
                logger.warning("Provider unhealthy", provider=provider_type.value)
                continue
            
            # Check capability
            if not provider.supports_job_type(job_type):
                continue
                
            # Check GPU if required
            if required_gpu and not provider.has_gpu(required_gpu):
                continue
            
            # Get metrics
            cost = await provider.get_cost_estimate(job_type, required_gpu)
            latency = await provider.get_latency_estimate(job_type)
            availability = await provider.get_availability()
            
            # Filter by constraints
            if max_cost_per_hour and cost > max_cost_per_hour:
                continue
            if max_latency_ms and latency > max_latency_ms:
                continue
            
            score = ProviderScore(
                provider=provider_type,
                cost_score=min(cost * 10, 100),  # Normalize to 0-100
                latency_score=min(latency / 100, 100),  # Normalize
                availability_score=availability,
            )
            score.calculate_total()
            scores.append(score)
        
        return sorted(scores, key=lambda s: s.total_score)
    
    def _apply_strategy(
        self,
        scores: list[ProviderScore],
        strategy: RoutingStrategy,
        prefer_provider: Optional[ProviderType] = None,
    ) -> ProviderScore:
        """Apply routing strategy to select provider."""
        if prefer_provider:
            for score in scores:
                if score.provider == prefer_provider:
                    return score
        
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            return min(scores, key=lambda s: s.cost_score)
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            return min(scores, key=lambda s: s.latency_score)
        elif strategy == RoutingStrategy.FAILOVER:
            # Primary is GCP for dev, RunPod for prod
            primary = ProviderType.RUNPOD if settings.is_production else ProviderType.GCP
            for score in scores:
                if score.provider == primary:
                    return score
            return scores[0]
        else:  # BALANCED
            return scores[0]  # Already sorted by total score
    
    def _get_fallback(
        self,
        scores: list[ProviderScore],
        selected: ProviderType,
    ) -> Optional[ProviderType]:
        """Get fallback provider if primary fails."""
        for score in scores:
            if score.provider != selected:
                return score.provider
        return None
    
    async def _check_budget(self) -> bool:
        """Check if we're within budget limits."""
        # Reset daily tracker if needed
        if datetime.now(timezone.utc) - self._last_cost_reset > timedelta(days=1):
            self._cost_tracker = {}
            self._last_cost_reset = datetime.now(timezone.utc)
        
        daily_spend = sum(self._cost_tracker.values())
        # Use Vast.ai config for budget (it has explicit budget settings)
        daily_limit = self._cloud_config.vastai.monthly_budget_usd / 30
        return daily_spend < daily_limit
    
    async def _route_budget_exceeded(self, _job_type: JobType) -> RoutingDecision:
        """Route when budget is exceeded - use GCP fallback."""
        return RoutingDecision(
            provider=ProviderType.GCP,
            reason="Budget exceeded, using GCP development fallback",
            estimated_cost=0.0,
            estimated_latency_ms=500,
            fallback_provider=None,
            metadata={"budget_exceeded": True}
        )
    
    def record_cost(self, provider: ProviderType, cost: float) -> None:
        """Record cost for budget tracking."""
        key = f"{provider.value}_{datetime.now(timezone.utc).date()}"
        self._cost_tracker[key] = self._cost_tracker.get(key, 0) + cost
        logger.info("Cost recorded", provider=provider.value, cost=cost)
    
    async def get_all_health(self) -> dict[str, bool]:
        """Get health status of all providers."""
        results = {}
        for provider_type, provider in self.providers.items():
            results[provider_type.value] = await provider.health_check()
        return results


# Singleton instance
_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    """Get or create the provider router singleton."""
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router
