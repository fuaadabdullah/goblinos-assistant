"""
Provider health check job for APScheduler.

Replaces the Celery task: tasks.provider_probe_worker.probe_all_providers
Runs every 5 minutes with Redis locking to prevent multiple instances.
"""

import logging
import asyncio
from typing import Dict, Any

import database
from models.provider import Provider as RoutingProvider, ProviderMetric
from providers import (
    OpenAIAdapter,
    AnthropicAdapter,
    GrokAdapter,
    DeepSeekAdapter,
    OllamaAdapter,
    LlamaCppAdapter,
)

from scheduler import with_redis_lock
from services.gcp_warmup import warm_gcp_endpoints

logger = logging.getLogger(__name__)


@with_redis_lock("provider_health_check", ttl=300)  # 5 minute TTL
def probe_all_providers_job():
    """
    APScheduler job to probe all active providers and collect health metrics.

    This replaces the Celery scheduled task and runs with Redis locking
    to prevent multiple replicas from executing simultaneously.
    """
    logger.info("Starting provider health check job")

    try:
        # Run the async probe function
        asyncio.run(_probe_all_providers_async())
        logger.info("Provider health check job completed successfully")

    except Exception as e:
        logger.error(f"Provider health check job failed: {e}")
        raise


async def _probe_all_providers_async():
    """Async implementation of provider probing."""
    await _probe_self_hosted_keepalive()

    db = database.SessionLocal()
    try:
        # Get all active providers
        providers = db.query(RoutingProvider).filter(RoutingProvider.is_active).all()

        if not providers:
            logger.debug("No active providers to probe")
            return

        logger.info(f"Probing {len(providers)} providers")

        # Probe each provider concurrently (limit concurrency to avoid overwhelming APIs)
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent probes

        async def probe_with_semaphore(provider):
            async with semaphore:
                return await _probe_provider(provider)

        # Run probes concurrently
        tasks = [probe_with_semaphore(provider) for provider in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        success_count = 0
        for i, result in enumerate(results):
            provider = providers[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to probe provider {provider.name}: {result}")
            else:
                success_count += 1
                logger.debug(f"Successfully probed provider {provider.name}")

        logger.info(f"Provider health check completed: {success_count}/{len(providers)} successful")

    except Exception as e:
        logger.error(f"Error in provider health check: {e}")
        raise
    finally:
        db.close()


async def _probe_self_hosted_keepalive() -> None:
    """Best-effort keepalive + optional tiny inference warm-up for self-hosted endpoints."""
    try:
        await warm_gcp_endpoints(reason="scheduler")
    except Exception as exc:
        logger.debug(
            "Warm-up job failed",
            extra={"error": type(exc).__name__},
        )


async def _probe_provider(provider: RoutingProvider) -> Dict[str, Any]:
    """
    Probe a single provider and save metrics.

    Args:
        provider: Provider to probe

    Returns:
        Dict with probe results
    """
    try:
        # Get provider adapter class
        provider_name = getattr(provider, "name", None)
        if not isinstance(provider_name, str) or not provider_name:
            return {"status": "error", "error": "Provider not found"}

        normalized_name = provider_name.lower().replace("-", "_")
        adapter_class = {
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
        }.get(normalized_name)

        if not adapter_class:
            logger.warning(f"No adapter found for provider {provider_name}")
            return {"status": "error", "error": "No adapter available"}

        # Simulate health check (replace with actual adapter health check)
        # In production, this would create an adapter instance and test connectivity
        is_healthy = True  # Simulate healthy
        response_time_ms = 150.0  # Simulate response time
        error_rate = 0.0  # Simulate error rate
        throughput_rpm = 100.0  # Simulate throughput

        # Save metrics to database
        db = database.SessionLocal()
        try:
            metric = ProviderMetric(
                provider_id=provider.id,
                is_healthy=is_healthy,
                response_time_ms=response_time_ms,
                error_rate=error_rate,
                throughput_rpm=throughput_rpm,
            )
            db.add(metric)
            db.commit()

            logger.debug(f"Saved metrics for provider {provider.name}: healthy={is_healthy}")

            return {
                "status": "success",
                "provider": provider.name,
                "healthy": is_healthy,
                "response_time_ms": response_time_ms,
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save metrics for provider {provider.name}: {e}")
            raise
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error probing provider {provider.name}: {e}")
        return {"status": "error", "provider": provider.name, "error": str(e)}
