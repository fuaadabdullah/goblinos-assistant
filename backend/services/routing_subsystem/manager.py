"""
Routing manager - main entry point for the routing subsystem.

Provides unified interface for routing decisions, health monitoring, and metrics.
"""

import asyncio
import inspect
import logging
import time
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from contextlib import asynccontextmanager

from .decision_engine import get_decision_engine, RoutingDecision
from .provider_health import get_provider_health_monitor
from .cache import get_routing_cache
from .policies import get_policy_manager

if TYPE_CHECKING:
    from backend.providers.base import InferenceRequest, InferenceResult
    from backend.providers.registry import get_provider_registry
else:
    try:
        from providers.base import InferenceRequest, InferenceResult
        from providers.registry import get_provider_registry
    except Exception:
        from backend.providers.base import InferenceRequest, InferenceResult
        from backend.providers.registry import get_provider_registry

logger = logging.getLogger(__name__)


class RoutingManager:
    """Main routing manager coordinating all routing components."""

    def __init__(self):
        """Initialize routing manager."""
        self.decision_engine = get_decision_engine()
        self.health_monitor = get_provider_health_monitor()
        self.cache = get_routing_cache()
        self.policy_manager = get_policy_manager()
        self.registry = get_provider_registry()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start the routing manager and background tasks."""
        logger.info("Starting routing manager")

        # Start health monitoring
        self.health_monitor.start_monitoring()

        # Start cache cleanup task
        cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        self._background_tasks.append(cleanup_task)

        logger.info("Routing manager started")

    async def stop(self):
        """Stop the routing manager and background tasks."""
        logger.info("Stopping routing manager")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop health monitoring
        self.health_monitor.stop_monitoring()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("Routing manager stopped")

    async def route_request(
        self,
        request: InferenceRequest,
        policy_name: str = "latency_first",
        use_cache: bool = True,
        client_key: Optional[str] = None,
    ) -> InferenceResult:
        """Route an inference request to the best provider.

        Args:
            request: Inference request
            policy_name: Routing policy to use
            use_cache: Whether to use decision caching
            client_key: Client identifier for rate limiting

        Returns:
            Inference result

        Raises:
            ValueError: If no providers available or request invalid
        """
        start_time = time.time()

        # Validate request
        validation_errors = self.decision_engine.validate_request(request)
        if validation_errors:
            raise ValueError(f"Invalid request: {', '.join(validation_errors)}")

        # Check rate limiting if client key provided
        if client_key:
            allowed, retry_after = self.cache.check_rate_limit(
                client_key, limit=100
            )  # 100 req/min
            if not allowed:
                raise ValueError(
                    f"Rate limit exceeded. Retry after {retry_after:.1f} seconds"
                )

        # Get routing decision
        try:
            decision = self.decision_engine.select_provider(
                request, policy_name, use_cache
            )
        except ValueError as e:
            logger.error(f"Routing decision failed: {e}")
            raise

        # Execute request with fallback logic
        result = await self._execute_with_fallback(request, decision)

        # Record metrics
        latency_ms = int((time.time() - start_time) * 1000)
        success = result.success if hasattr(result, "success") else True

        self.cache.record_provider_request(decision.provider_id, latency_ms, success)
        self.health_monitor.record_inference_metrics(
            provider_id=decision.provider_id,
            latency_ms=latency_ms,
            success=success,
            request=request,
            result=result,
        )

        # Emit routing metrics
        self._emit_routing_metrics(decision, latency_ms, success, policy_name)

        return result

    async def _execute_with_fallback(
        self, request: InferenceRequest, decision: RoutingDecision
    ) -> InferenceResult:
        """Execute request with fallback providers.

        Args:
            request: Inference request
            decision: Routing decision with fallbacks

        Returns:
            Inference result
        """
        providers_to_try = [decision.provider_id] + decision.fallback_providers

        for provider_id in providers_to_try:
            try:
                provider = self.registry.get_provider(provider_id)
                if not provider:
                    logger.warning(f"Provider {provider_id} not found in registry")
                    continue

                logger.info(f"Attempting request with provider: {provider_id}")

                # Handle both sync and async infer methods
                infer_result = provider.infer(request)
                if inspect.iscoroutine(infer_result):
                    result = await infer_result
                else:
                    result = infer_result

                if result.success:
                    return result
                else:
                    logger.warning(
                        f"Provider {provider_id} failed: {result.error_message}"
                    )

            except Exception as e:
                logger.warning(f"Provider {provider_id} error: {e}")
                continue

        # All providers failed
        return InferenceResult(
            content="",
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            model=request.model if hasattr(request, "model") else "unknown",
            success=False,
            error_message="All providers failed",
            latency_ms=0,
        )

    def get_provider_rankings(
        self, request: InferenceRequest, policy_name: str = "latency_first"
    ) -> List[tuple]:
        """Get ranked list of providers for a request.

        Args:
            request: Inference request
            policy_name: Routing policy

        Returns:
            List of (provider_id, score) tuples
        """
        return self.decision_engine.get_provider_rankings(request, policy_name)

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status.

        Returns:
            System status dictionary
        """
        available_providers = self.registry.get_available_providers()
        provider_statuses = {}

        for provider in available_providers:
            provider_id = provider.provider_id
            health_info = self.health_monitor.get_provider_health(provider_id)
            metrics = self.cache.get_provider_metrics(provider_id)

            provider_statuses[provider_id] = {
                "health": health_info.get("health_status", "unknown"),
                "metrics": metrics,
            }

        return {
            "total_providers": len(available_providers),
            "healthy_providers": sum(
                1
                for status in provider_statuses.values()
                if status["health"] == "healthy"
            ),
            "providers": provider_statuses,
            "policies": list(self.policy_manager.get_active_policies()),
        }

    def _emit_routing_metrics(
        self,
        decision: RoutingDecision,
        latency_ms: int,
        success: bool,
        policy_name: str,
    ):
        """Emit routing-related metrics.

        Args:
            decision: Routing decision
            latency_ms: Total routing latency
            success: Whether request succeeded
            policy_name: Policy used
        """
        # This would integrate with your metrics system
        # For now, just log
        logger.info(
            f"Routing metrics - provider: {decision.provider_id}, "
            f"policy: {policy_name}, latency: {latency_ms}ms, "
            f"success: {success}, cache_hit: {decision.cache_hit}"
        )

    async def _cache_cleanup_loop(self):
        """Background task to clean up expired cache entries."""
        while not self._shutdown_event.is_set():
            try:
                self.cache.cleanup_expired_cache()
                await asyncio.sleep(300)  # Clean every 5 minutes
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)  # Retry sooner on error


# Global routing manager instance
_manager: Optional[RoutingManager] = None


def get_routing_manager() -> RoutingManager:
    """Get the global routing manager instance."""
    global _manager
    if _manager is None:
        _manager = RoutingManager()
    assert _manager is not None
    return _manager


@asynccontextmanager
async def routing_lifecycle():
    """Context manager for routing manager lifecycle."""
    manager = get_routing_manager()
    await manager.start()
    try:
        yield manager
    finally:
        await manager.stop()
