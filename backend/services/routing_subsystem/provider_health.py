"""
Provider health monitoring and metrics ingestion.

Monitors provider health, collects metrics, and provides health status
for routing decisions.
"""

import time
import asyncio
import threading
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from backend.providers.registry import get_provider_registry
else:
    try:
        from providers.registry import get_provider_registry
    except Exception:
        from backend.providers.registry import get_provider_registry

logger = logging.getLogger(__name__)


class ProviderHealthMonitor:
    """Monitors provider health and collects performance metrics."""

    def __init__(self, check_interval: int = 60):
        """Initialize health monitor.

        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        self.provider_metrics: Dict[str, Dict[str, Any]] = {}
        self.last_check: Dict[str, float] = {}
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self):
        """Start background health monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started provider health monitoring")

    def stop_monitoring(self):
        """Stop background health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Stopped provider health monitoring")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                asyncio.run(self._perform_health_checks())
            except Exception as e:
                logger.error(f"Health check error: {e}")

            time.sleep(self.check_interval)

    async def _perform_health_checks(self):
        """Perform health checks on all providers."""
        registry = get_provider_registry()
        providers = registry.get_available_providers()

        for provider in providers:
            try:
                await self._check_provider_health(provider)
            except Exception as e:
                logger.warning(
                    f"Failed to check health for {provider.provider_id}: {e}"
                )

    async def _check_provider_health(self, provider):
        """Check health of a single provider."""
        provider_id = provider.provider_id
        start_time = time.time()

        try:
            # Perform health check
            health_status = provider.health_check()

            check_time = time.time() - start_time
            self.last_check[provider_id] = time.time()

            # Update metrics
            if provider_id not in self.provider_metrics:
                self.provider_metrics[provider_id] = {
                    "checks_total": 0,
                    "checks_success": 0,
                    "avg_check_time": 0,
                    "last_success": None,
                    "last_failure": None,
                    "consecutive_failures": 0,
                    "health_status": "unknown",
                }

            metrics = self.provider_metrics[provider_id]
            metrics["checks_total"] += 1
            metrics["health_status"] = health_status.value

            if health_status.name == "HEALTHY":
                metrics["checks_success"] += 1
                metrics["last_success"] = time.time()
                metrics["consecutive_failures"] = 0
            else:
                metrics["last_failure"] = time.time()
                metrics["consecutive_failures"] += 1

            # Update average check time
            metrics["avg_check_time"] = (
                (metrics["avg_check_time"] * (metrics["checks_total"] - 1)) + check_time
            ) / metrics["checks_total"]

            logger.debug(
                f"Health check for {provider_id}: {health_status.value} ({check_time:.2f}s)"
            )

        except Exception as e:
            logger.warning(f"Health check failed for {provider_id}: {e}")

            # Record failure
            if provider_id not in self.provider_metrics:
                self.provider_metrics[provider_id] = {
                    "checks_total": 0,
                    "checks_success": 0,
                    "avg_check_time": 0,
                    "last_success": None,
                    "last_failure": None,
                    "consecutive_failures": 0,
                    "health_status": "unknown",
                }

            metrics = self.provider_metrics[provider_id]
            metrics["last_failure"] = time.time()
            metrics["consecutive_failures"] += 1
            metrics["health_status"] = "unhealthy"

    def get_provider_health(self, provider_id: str) -> Dict[str, Any]:
        """Get health information for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            Health metrics dictionary
        """
        return self.provider_metrics.get(provider_id, {})

    def get_all_provider_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health information for all providers.

        Returns:
            Dictionary mapping provider IDs to health metrics
        """
        return self.provider_metrics.copy()

    def is_provider_healthy(self, provider_id: str, max_age: int = 300) -> bool:
        """Check if a provider is currently healthy.

        Args:
            provider_id: Provider identifier
            max_age: Maximum age of health check in seconds

        Returns:
            True if provider is healthy and check is recent
        """
        metrics = self.get_provider_health(provider_id)
        if not metrics:
            return False

        # Check if health check is recent enough
        last_check = self.last_check.get(provider_id, 0)
        if time.time() - last_check > max_age:
            return False

        return metrics.get("health_status") == "healthy"

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary.

        Returns:
            Summary of provider health status
        """
        total_providers = len(self.provider_metrics)
        healthy_providers = sum(
            1
            for metrics in self.provider_metrics.values()
            if metrics.get("health_status") == "healthy"
        )

        return {
            "total_providers": total_providers,
            "healthy_providers": healthy_providers,
            "unhealthy_providers": total_providers - healthy_providers,
            "overall_health": healthy_providers / total_providers
            if total_providers > 0
            else 0,
            "last_updated": max(self.last_check.values()) if self.last_check else None,
        }

    def record_inference_metrics(
        self, provider_id: str, latency_ms: int, success: bool, **extra
    ):
        """Record inference metrics for a provider.

        Args:
            provider_id: Provider identifier
            latency_ms: Inference latency in milliseconds
            success: Whether the inference was successful
            **extra: Additional metrics
        """
        if provider_id not in self.provider_metrics:
            self.provider_metrics[provider_id] = {}

        metrics = self.provider_metrics[provider_id]

        # Ensure inference metrics exist (may be missing if only health checks ran before)
        if "inferences_total" not in metrics:
            metrics["inferences_total"] = 0
            metrics["inferences_success"] = 0
            metrics["avg_latency_ms"] = 0
            metrics["last_inference"] = None
            metrics["error_rate"] = 0

        metrics["inferences_total"] += 1
        metrics["last_inference"] = time.time()

        if success:
            metrics["inferences_success"] += 1
        else:
            # Update error rate
            error_rate = (
                metrics["inferences_total"] - metrics["inferences_success"]
            ) / metrics["inferences_total"]
            metrics["error_rate"] = error_rate

        # Update average latency
        metrics["avg_latency_ms"] = (
            (metrics["avg_latency_ms"] * (metrics["inferences_total"] - 1)) + latency_ms
        ) / metrics["inferences_total"]

        # Store additional metrics
        for key, value in extra.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)

            # Keep only last 100 values for rolling averages
            if len(metrics[key]) > 100:
                metrics[key] = metrics[key][-100:]


# Global health monitor instance
_health_monitor: Optional[ProviderHealthMonitor] = None


def get_provider_health_monitor() -> ProviderHealthMonitor:
    """Get the global provider health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ProviderHealthMonitor()
        _health_monitor.start_monitoring()
    assert _health_monitor is not None
    return _health_monitor
