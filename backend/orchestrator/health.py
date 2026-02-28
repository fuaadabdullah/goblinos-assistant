"""
Health Monitoring Module
Monitors provider health and availability for intelligent routing.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Callable, Any

from .providers.base import ProviderType, BaseProvider

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Provider health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    provider: ProviderType
    status: HealthStatus
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict = field(default_factory=dict)


@dataclass
class ProviderHealth:
    """Aggregated health state for a provider"""
    provider: ProviderType
    status: HealthStatus
    availability_percent: float  # Last 24h availability
    avg_latency_ms: float
    last_check: datetime
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    
    @property
    def is_available(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


class HealthMonitor:
    """
    Monitors health and availability of all providers.
    
    Performs periodic health checks, tracks availability metrics,
    and provides health status for routing decisions.
    """
    
    # Health check configuration
    CHECK_INTERVAL_SECONDS = 60
    UNHEALTHY_THRESHOLD = 3  # Consecutive failures before marking unhealthy
    DEGRADED_LATENCY_MS = 2000  # Latency threshold for degraded status
    HISTORY_RETENTION_HOURS = 24
    
    def __init__(self):
        self._providers: dict[ProviderType, BaseProvider] = {}
        self._health_history: dict[ProviderType, list[HealthCheckResult]] = {}
        self._current_status: dict[ProviderType, ProviderHealth] = {}
        self._check_task: Optional[asyncio.Task] = None
        self._callbacks: list[Callable[[ProviderType, HealthStatus], Any]] = []
    
    def register_provider(self, provider_type: ProviderType, provider: BaseProvider) -> None:
        """Register a provider for health monitoring"""
        self._providers[provider_type] = provider
        self._health_history[provider_type] = []
        self._current_status[provider_type] = ProviderHealth(
            provider=provider_type,
            status=HealthStatus.UNKNOWN,
            availability_percent=100.0,
            avg_latency_ms=0.0,
            last_check=datetime.now(timezone.utc),
        )
    
    def register_callback(self, callback: Callable[[ProviderType, HealthStatus], Any]) -> None:
        """Register callback for health status changes"""
        self._callbacks.append(callback)
    
    async def start(self) -> None:
        """Start the health monitoring loop"""
        if self._check_task is None or self._check_task.done():
            self._check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Health monitoring started")
    
    async def stop(self) -> None:
        """Stop the health monitoring loop"""
        if self._check_task and not self._check_task.done():
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            logger.info("Health monitoring stopped")
    
    async def _health_check_loop(self) -> None:
        """Main health check loop"""
        while True:
            try:
                await self.check_all_providers()
                await asyncio.sleep(self.CHECK_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.CHECK_INTERVAL_SECONDS)
    
    async def check_all_providers(self) -> dict[ProviderType, HealthCheckResult]:
        """Check health of all registered providers"""
        results = {}
        
        tasks = [
            self._check_provider(provider_type, provider)
            for provider_type, provider in self._providers.items()
        ]
        
        if tasks:
            check_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in check_results:
                if isinstance(result, HealthCheckResult):
                    results[result.provider] = result
        
        return results
    
    async def _check_provider(
        self,
        provider_type: ProviderType,
        provider: BaseProvider
    ) -> HealthCheckResult:
        """Check health of a single provider"""
        start_time = datetime.now(timezone.utc)
        
        try:
            is_healthy = await provider.health_check()
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            if is_healthy:
                status = HealthStatus.HEALTHY if latency_ms < self.DEGRADED_LATENCY_MS else HealthStatus.DEGRADED
                error = None
            else:
                status = HealthStatus.UNHEALTHY
                error = "Health check returned false"
            
            result = HealthCheckResult(
                provider=provider_type,
                status=status,
                latency_ms=latency_ms,
                error=error,
            )
            
        except Exception as e:
            result = HealthCheckResult(
                provider=provider_type,
                status=HealthStatus.UNHEALTHY,
                error=str(e),
            )
        
        # Update history and status
        self._update_health_state(result)
        
        return result
    
    def _update_health_state(self, result: HealthCheckResult) -> None:
        """Update health history and current status"""
        provider = result.provider
        
        # Add to history
        self._health_history[provider].append(result)
        
        # Trim old history
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.HISTORY_RETENTION_HOURS)
        self._health_history[provider] = [
            r for r in self._health_history[provider]
            if r.timestamp > cutoff
        ]
        
        # Calculate availability
        history = self._health_history[provider]
        if history:
            healthy_count = sum(1 for r in history if r.status == HealthStatus.HEALTHY)
            availability = (healthy_count / len(history)) * 100
            
            # Calculate average latency
            latencies = [r.latency_ms for r in history if r.latency_ms is not None]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        else:
            availability = 100.0
            avg_latency = 0.0
        
        # Get previous status for callback
        prev_status = self._current_status[provider].status
        
        # Determine current status based on consecutive failures
        current = self._current_status[provider]
        
        if result.status == HealthStatus.UNHEALTHY:
            current.consecutive_failures += 1
            current.last_error = result.error
            
            if current.consecutive_failures >= self.UNHEALTHY_THRESHOLD:
                new_status = HealthStatus.UNHEALTHY
            else:
                new_status = HealthStatus.DEGRADED
        else:
            current.consecutive_failures = 0
            current.last_error = None
            new_status = result.status
        
        # Update current status
        self._current_status[provider] = ProviderHealth(
            provider=provider,
            status=new_status,
            availability_percent=availability,
            avg_latency_ms=avg_latency,
            last_check=result.timestamp,
            consecutive_failures=current.consecutive_failures,
            last_error=current.last_error,
        )
        
        # Trigger callbacks on status change
        if new_status != prev_status:
            self._notify_status_change(provider, new_status)
    
    def _notify_status_change(self, provider: ProviderType, status: HealthStatus) -> None:
        """Notify callbacks of status change"""
        logger.info(f"Provider {provider.value} status changed to {status.value}")
        
        for callback in self._callbacks:
            try:
                result = callback(provider, status)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"Health callback error: {e}")
    
    def get_provider_health(self, provider: ProviderType) -> Optional[ProviderHealth]:
        """Get current health status for a provider"""
        return self._current_status.get(provider)
    
    def get_all_health(self) -> dict[ProviderType, ProviderHealth]:
        """Get health status for all providers"""
        return self._current_status.copy()
    
    def get_available_providers(self) -> list[ProviderType]:
        """Get list of currently available providers"""
        return [
            p for p, h in self._current_status.items()
            if h.is_available
        ]
    
    def get_healthiest_provider(self) -> Optional[ProviderType]:
        """Get the healthiest provider (for fallback selection)"""
        available = [
            (p, h) for p, h in self._current_status.items()
            if h.status == HealthStatus.HEALTHY
        ]
        
        if not available:
            # Fall back to degraded providers
            available = [
                (p, h) for p, h in self._current_status.items()
                if h.status == HealthStatus.DEGRADED
            ]
        
        if not available:
            return None
        
        # Sort by availability and latency
        available.sort(key=lambda x: (-x[1].availability_percent, x[1].avg_latency_ms))
        
        return available[0][0]
    
    def get_health_summary(self) -> dict:
        """Get summary of all provider health"""
        return {
            provider.value: {
                "status": health.status.value,
                "availability_percent": health.availability_percent,
                "avg_latency_ms": health.avg_latency_ms,
                "consecutive_failures": health.consecutive_failures,
                "last_check": health.last_check.isoformat(),
                "last_error": health.last_error,
            }
            for provider, health in self._current_status.items()
        }
    
    async def force_check(self, provider: ProviderType) -> Optional[HealthCheckResult]:
        """Force an immediate health check for a provider"""
        if provider not in self._providers:
            return None
        
        return await self._check_provider(provider, self._providers[provider])
