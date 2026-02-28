"""
Background worker for collecting provider health metrics and performance data.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from ..models.routing import RoutingProvider, ProviderMetric

from ..providers import (
    OpenAIAdapter,
    AnthropicAdapter,
    GrokAdapter,
    DeepSeekAdapter,
    OllamaAdapter,
    LlamaCppAdapter,
)
from ..services.encryption import EncryptionService
from ..database import SessionLocal

logger = logging.getLogger(__name__)


class ProviderProbeWorker:
    """Background worker that continuously monitors provider health and collects metrics."""

    def __init__(self, encryption_key: str, probe_interval: int = 300):
        """Initialize the probe worker.

        Args:
            encryption_key: Key for decrypting API keys
            probe_interval: Seconds between health checks (default: 5 minutes)
        """
        self.encryption_service = EncryptionService(encryption_key)
        self.probe_interval = probe_interval
        self.adapters = {
            "openai": OpenAIAdapter,
            "aliyun": OpenAIAdapter,
            "alibaba": OpenAIAdapter,
            "alibaba_cloud": OpenAIAdapter,
            "azure": OpenAIAdapter,
            "azure_openai": OpenAIAdapter,
            "ollama": OllamaAdapter,
            "ollama_gcp": OllamaAdapter,
            "llamacpp": LlamaCppAdapter,
            "llamacpp_gcp": LlamaCppAdapter,
            "anthropic": AnthropicAdapter,
            "grok": GrokAdapter,
            "deepseek": DeepSeekAdapter,
        }
        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background probing task."""
        if self.running:
            logger.warning("Provider probe worker is already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._probe_loop())
        logger.info(
            f"Started provider probe worker with {self.probe_interval}s interval"
        )

    async def stop(self) -> None:
        """Stop the background probing task."""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped provider probe worker")

    async def _probe_loop(self) -> None:
        """Main probing loop that runs indefinitely."""
        while self.running:
            try:
                await self._probe_all_providers()
            except Exception as e:
                logger.error(f"Error in probe loop: {e}")

            # Wait for next probe interval
            await asyncio.sleep(self.probe_interval)

    async def _probe_all_providers(self) -> None:
        """Probe all active providers and collect metrics."""
        db = SessionLocal()
        try:
            # Get all active providers
            providers = (
                db.query(RoutingProvider).filter(RoutingProvider.is_active).all()
            )

            if not providers:
                logger.debug("No active providers to probe")
                return

            logger.debug(f"Probing {len(providers)} providers")

            # Probe each provider concurrently
            tasks = []
            for provider in providers:
                task = asyncio.create_task(self._probe_provider(provider))
                tasks.append(task)

            # Wait for all probes to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and save metrics
            for provider, result in zip(providers, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to probe provider {provider.name}: {result}")
                    await self._save_metric(db, provider.id, False, error=str(result))
                else:
                    await self._save_metric(db, provider.id, **result)

            db.commit()

        except Exception as e:
            logger.error(f"Error probing providers: {e}")
            db.rollback()
        finally:
            db.close()

    async def _probe_provider(self, provider: RoutingProvider) -> Dict[str, Any]:
        """Probe a single provider for health and performance metrics.

        Args:
            provider: Provider to probe

        Returns:
            Dict with health metrics
        """
        try:
            # Get API key - try encrypted first, fall back to plain text
            api_key = None
            if provider.api_key_encrypted:
                try:
                    api_key = self.encryption_service.decrypt(
                        provider.api_key_encrypted
                    )
                except Exception:
                    pass
            if not api_key and provider.api_key:
                api_key = provider.api_key
            if not api_key:
                raise ValueError(f"No API key available for provider {provider.name}")

            # Get adapter
            adapter_class = self.adapters.get(provider.name.lower().replace("-", "_"))
            if not adapter_class:
                raise ValueError(f"No adapter found for provider {provider.name}")

            # Initialize adapter
            adapter = adapter_class(api_key, provider.base_url)

            # Perform health check
            health_result = await adapter.health_check()

            # Test completion if health check passed
            completion_result = None
            if health_result.get("healthy", False):
                try:
                    completion_result = await adapter.test_completion()
                except Exception as e:
                    logger.warning(f"Completion test failed for {provider.name}: {e}")

            # Combine results
            metrics = {
                "is_healthy": health_result.get("healthy", False),
                "response_time_ms": health_result.get("response_time_ms"),
                "error_rate": health_result.get("error_rate", 0.0),
                "throughput_rpm": health_result.get("available_models", 0)
                * 10,  # Estimate based on model count
                "tokens_used": completion_result.get("tokens_used")
                if completion_result
                else None,
                "metadata": {
                    "available_models": health_result.get("available_models", 0),
                    "completion_test_success": completion_result.get("success")
                    if completion_result
                    else False,
                    "completion_response_time": completion_result.get(
                        "response_time_ms"
                    )
                    if completion_result
                    else None,
                },
            }

            return metrics

        except Exception as e:
            logger.error(f"Error probing provider {provider.name}: {e}")
            return {
                "is_healthy": False,
                "response_time_ms": None,
                "error_rate": 1.0,
                "throughput_rpm": 0,
                "tokens_used": None,
                "metadata": {"error": str(e)},
            }

    async def _save_metric(
        self,
        db: Session,
        provider_id: int,
        is_healthy: bool,
        response_time_ms: Optional[float] = None,
        error_rate: float = 0.0,
        throughput_rpm: Optional[float] = None,
        tokens_used: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Save provider metrics to database.

        Args:
            db: Database session
            provider_id: Provider ID
            is_healthy: Whether provider is healthy
            response_time_ms: Response time in milliseconds
            error_rate: Error rate (0.0 to 1.0)
            throughput_rpm: Throughput in requests per minute
            tokens_used: Tokens used in test
            metadata: Additional metadata
            error: Error message if any
        """
        try:
            # Calculate cost if tokens used (rough estimate)
            cost_incurred = None
            if tokens_used:
                # Get provider for cost calculation
                provider = (
                    db.query(RoutingProvider)
                    .filter(RoutingProvider.id == provider_id)
                    .first()
                )
                if provider and provider.cost_per_token:
                    cost_incurred = tokens_used * provider.cost_per_token

            # Create metric record
            metric = ProviderMetric(
                provider_id=provider_id,
                is_healthy=is_healthy,
                response_time_ms=response_time_ms,
                error_rate=error_rate,
                throughput_rpm=throughput_rpm,
                tokens_used=tokens_used,
                cost_incurred=cost_incurred,
                metadata=metadata or {},
            )

            db.add(metric)

            # Clean up old metrics (keep last 1000 per provider)
            old_metrics = (
                db.query(ProviderMetric)
                .filter(ProviderMetric.provider_id == provider_id)
                .order_by(ProviderMetric.timestamp.desc())
                .offset(1000)
                .all()
            )

            for old_metric in old_metrics:
                db.delete(old_metric)

            logger.debug(
                f"Saved metrics for provider {provider_id}: healthy={is_healthy}"
            )

        except Exception as e:
            logger.error(f"Failed to save metrics for provider {provider_id}: {e}")

    async def get_provider_status(self, provider_id: int) -> Dict[str, Any]:
        """Get current status and recent metrics for a provider.

        Args:
            provider_id: Provider ID

        Returns:
            Dict with provider status and metrics
        """
        db = SessionLocal()
        try:
            # Get latest metric
            latest_metric = (
                db.query(ProviderMetric)
                .filter(ProviderMetric.provider_id == provider_id)
                .order_by(ProviderMetric.timestamp.desc())
                .first()
            )

            if not latest_metric:
                return {"status": "unknown", "metrics": None}

            # Get recent metrics for trends
            recent_metrics = (
                db.query(ProviderMetric)
                .filter(ProviderMetric.provider_id == provider_id)
                .order_by(ProviderMetric.timestamp.desc())
                .limit(10)
                .all()
            )

            # Calculate averages
            avg_response_time = None
            avg_error_rate = 0.0
            if recent_metrics:
                response_times = [
                    m.response_time_ms for m in recent_metrics if m.response_time_ms
                ]
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)

                error_rates = [
                    m.error_rate for m in recent_metrics if m.error_rate is not None
                ]
                if error_rates:
                    avg_error_rate = sum(error_rates) / len(error_rates)

            return {
                "status": "healthy" if latest_metric.is_healthy else "unhealthy",
                "last_check": latest_metric.timestamp.isoformat(),
                "metrics": {
                    "response_time_ms": latest_metric.response_time_ms,
                    "error_rate": latest_metric.error_rate,
                    "throughput_rpm": latest_metric.throughput_rpm,
                    "avg_response_time_ms": avg_response_time,
                    "avg_error_rate": avg_error_rate,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get provider status: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            db.close()
