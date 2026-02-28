"""
Cost Optimizer Module
Optimizes provider selection based on cost efficiency and budget constraints.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from .providers.base import ProviderType, JobType

logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """Cost optimization strategies"""
    LOWEST_COST = "lowest_cost"
    BEST_VALUE = "best_value"  # Balance cost and performance
    BUDGET_PRESERVING = "budget_preserving"
    PERFORMANCE_FIRST = "performance_first"


@dataclass
class CostEstimate:
    """Cost estimate for a provider/job combination"""
    provider: ProviderType
    estimated_cost_usd: float
    estimated_duration_seconds: float
    confidence: float = 0.8
    breakdown: dict = field(default_factory=dict)


@dataclass
class BudgetStatus:
    """Current budget status"""
    daily_budget_usd: float
    daily_spent_usd: float
    monthly_budget_usd: float
    monthly_spent_usd: float
    daily_remaining_usd: float = 0.0
    monthly_remaining_usd: float = 0.0
    is_over_budget: bool = False
    
    def __post_init__(self):
        self.daily_remaining_usd = max(0, self.daily_budget_usd - self.daily_spent_usd)
        self.monthly_remaining_usd = max(0, self.monthly_budget_usd - self.monthly_spent_usd)
        self.is_over_budget = self.daily_remaining_usd <= 0 or self.monthly_remaining_usd <= 0


class CostOptimizer:
    """
    Optimizes provider selection based on cost analysis.
    
    Tracks spending, estimates costs, and recommends providers
    based on configured budgets and optimization strategies.
    """
    
    # Provider cost estimates per GPU-hour (USD)
    PROVIDER_COSTS = {
        ProviderType.GCP: {
            "rtx_4090": 0.0,  # Self-hosted, only compute costs
            "rtx_3090": 0.0,
            "default": 0.10,  # Minimal GCE costs
        },
        ProviderType.RUNPOD: {
            "rtx_4090": 0.44,
            "rtx_3090": 0.30,
            "a100_40gb": 0.79,
            "a100_80gb": 1.19,
            "default": 0.40,
        },
        ProviderType.VASTAI: {
            "rtx_4090": 0.35,  # Spot pricing, varies
            "rtx_3090": 0.25,
            "a100_40gb": 0.70,
            "default": 0.30,
        },
    }
    
    # Estimated tokens per second by provider
    PROVIDER_THROUGHPUT = {
        ProviderType.GCP: 30,  # Ollama on modest hardware
        ProviderType.RUNPOD: 80,  # RTX 4090 serverless
        ProviderType.VASTAI: 60,  # Variable spot instances
    }
    
    def __init__(
        self,
        daily_budget_usd: float = 50.0,
        monthly_budget_usd: float = 1500.0,
        strategy: OptimizationStrategy = OptimizationStrategy.BEST_VALUE
    ):
        self.daily_budget_usd = daily_budget_usd
        self.monthly_budget_usd = monthly_budget_usd
        self.strategy = strategy
        
        # Spending trackers
        self._daily_costs: dict[str, float] = {}
        self._monthly_costs: dict[str, float] = {}
        self._last_reset_date: datetime = datetime.now(timezone.utc)
    
    def estimate_cost(
        self,
        provider: ProviderType,
        job_type: JobType,
        input_tokens: int,
        output_tokens: int,
        gpu_type: str = "default"
    ) -> CostEstimate:
        """
        Estimate cost for a job on a specific provider.
        
        Args:
            provider: Target provider
            job_type: Type of job
            input_tokens: Expected input tokens
            output_tokens: Expected output tokens
            gpu_type: GPU type (affects pricing)
            
        Returns:
            CostEstimate with breakdown
        """
        total_tokens = input_tokens + output_tokens
        throughput = self.PROVIDER_THROUGHPUT.get(provider, 50)
        
        # Estimate duration
        duration_seconds = total_tokens / throughput
        duration_hours = duration_seconds / 3600
        
        # Get hourly rate
        provider_costs = self.PROVIDER_COSTS.get(provider, {})
        hourly_rate = provider_costs.get(gpu_type, provider_costs.get("default", 0.50))
        
        # Calculate base cost
        base_cost = duration_hours * hourly_rate
        
        # Add overhead for job type
        overhead_multiplier = {
            JobType.INFERENCE: 1.0,
            JobType.BATCH_INFERENCE: 0.9,  # Batching is more efficient
            JobType.TRAINING: 1.2,  # Extra overhead
            JobType.FINETUNING: 1.3,
        }.get(job_type, 1.0)
        
        estimated_cost = base_cost * overhead_multiplier
        
        return CostEstimate(
            provider=provider,
            estimated_cost_usd=round(estimated_cost, 6),
            estimated_duration_seconds=duration_seconds,
            confidence=0.8,
            breakdown={
                "base_cost": base_cost,
                "hourly_rate": hourly_rate,
                "duration_hours": duration_hours,
                "overhead_multiplier": overhead_multiplier,
                "tokens": total_tokens,
            }
        )
    
    def get_cheapest_provider(
        self,
        job_type: JobType,
        input_tokens: int,
        output_tokens: int,
        available_providers: list[ProviderType]
    ) -> tuple[ProviderType, CostEstimate]:
        """
        Find the cheapest provider for a job.
        
        Args:
            job_type: Type of job
            input_tokens: Input token count
            output_tokens: Output token count
            available_providers: List of available providers
            
        Returns:
            Tuple of (provider, cost_estimate)
        """
        estimates = []
        
        for provider in available_providers:
            estimate = self.estimate_cost(
                provider=provider,
                job_type=job_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            estimates.append((provider, estimate))
        
        # Sort by cost
        estimates.sort(key=lambda x: x[1].estimated_cost_usd)
        
        return estimates[0] if estimates else (ProviderType.GCP, CostEstimate(
            provider=ProviderType.GCP,
            estimated_cost_usd=0.0,
            estimated_duration_seconds=0.0
        ))
    
    def get_best_value_provider(
        self,
        job_type: JobType,
        input_tokens: int,
        output_tokens: int,
        available_providers: list[ProviderType],
        max_latency_seconds: Optional[float] = None
    ) -> tuple[ProviderType, CostEstimate]:
        """
        Find the best value provider (balancing cost and performance).
        
        Args:
            job_type: Type of job
            input_tokens: Input token count
            output_tokens: Output token count
            available_providers: List of available providers
            max_latency_seconds: Optional latency constraint
            
        Returns:
            Tuple of (provider, cost_estimate)
        """
        estimates = []
        
        for provider in available_providers:
            estimate = self.estimate_cost(
                provider=provider,
                job_type=job_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            # Filter by latency constraint
            if max_latency_seconds and estimate.estimated_duration_seconds > max_latency_seconds:
                continue
            
            # Calculate value score (lower is better)
            # Combine cost and time with weights
            value_score = (
                estimate.estimated_cost_usd * 0.6 +
                (estimate.estimated_duration_seconds / 60) * 0.4  # Normalize time to minutes
            )
            
            estimates.append((provider, estimate, value_score))
        
        if not estimates:
            # Fall back to cheapest without latency constraint
            return self.get_cheapest_provider(
                job_type, input_tokens, output_tokens, available_providers
            )
        
        # Sort by value score
        estimates.sort(key=lambda x: x[2])
        
        return estimates[0][0], estimates[0][1]
    
    def record_cost(self, provider: ProviderType, cost_usd: float) -> None:
        """Record actual cost for tracking"""
        self._check_and_reset_daily()
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        
        key = f"{provider.value}_{today}"
        self._daily_costs[key] = self._daily_costs.get(key, 0.0) + cost_usd
        
        month_key = f"{provider.value}_{month}"
        self._monthly_costs[month_key] = self._monthly_costs.get(month_key, 0.0) + cost_usd
        
        logger.debug(f"Recorded cost: {provider.value} ${cost_usd:.4f}")
    
    def get_budget_status(self) -> BudgetStatus:
        """Get current budget status"""
        self._check_and_reset_daily()
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        
        daily_spent = sum(
            v for k, v in self._daily_costs.items()
            if k.endswith(today)
        )
        
        monthly_spent = sum(
            v for k, v in self._monthly_costs.items()
            if k.endswith(month)
        )
        
        return BudgetStatus(
            daily_budget_usd=self.daily_budget_usd,
            daily_spent_usd=daily_spent,
            monthly_budget_usd=self.monthly_budget_usd,
            monthly_spent_usd=monthly_spent,
        )
    
    def can_afford(self, estimated_cost_usd: float) -> bool:
        """Check if we can afford a job within budget"""
        status = self.get_budget_status()
        return (
            estimated_cost_usd <= status.daily_remaining_usd and
            estimated_cost_usd <= status.monthly_remaining_usd
        )
    
    def _check_and_reset_daily(self) -> None:
        """Reset daily costs if day has changed"""
        now = datetime.now(timezone.utc)
        if now.date() > self._last_reset_date.date():
            # Keep only current day's costs
            today = now.strftime("%Y-%m-%d")
            self._daily_costs = {
                k: v for k, v in self._daily_costs.items()
                if k.endswith(today)
            }
            self._last_reset_date = now
    
    def get_cost_summary(self) -> dict:
        """Get summary of costs by provider"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        
        summary = {
            "daily": {},
            "monthly": {},
            "budget_status": self.get_budget_status().__dict__,
        }
        
        for provider in ProviderType:
            daily_key = f"{provider.value}_{today}"
            monthly_key = f"{provider.value}_{month}"
            
            summary["daily"][provider.value] = self._daily_costs.get(daily_key, 0.0)
            summary["monthly"][provider.value] = self._monthly_costs.get(monthly_key, 0.0)
        
        return summary
