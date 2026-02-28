"""
Minimal wrapper example for migrating provider probe from Celery to lightweight patterns.

Old Celery task:
    @celery.task
    def probe_provider(provider_id):
        # do network checks...

New patterns for tiny tasks:
    - APScheduler job: For in-app periodic scheduling
    - CronJob wrapper: For containerized execution
"""

import logging
import asyncio

from database import SessionLocal
from models.provider import Provider as RoutingProvider
from jobs.provider_health import _probe_provider
from scheduler import redis_lock

logger = logging.getLogger(__name__)


# ===== MINIMAL WRAPPER FUNCTION =====
# This replaces the old Celery task logic
async def probe_provider_minimal(provider_id: int) -> dict:
    """
    Minimal wrapper for probing a single provider.

    Replaces: @celery.task def probe_provider(provider_id)

    Args:
        provider_id: ID of the provider to probe

    Returns:
        Dict with probe results
    """
    db = SessionLocal()
    try:
        # Get provider from database
        provider = (
            db.query(RoutingProvider).filter(RoutingProvider.id == provider_id).first()
        )
        if not provider:
            return {"status": "error", "error": f"Provider {provider_id} not found"}

        # Use existing probe logic
        result = await _probe_provider(provider)
        return result

    except Exception as e:
        logger.error(f"Error in minimal probe wrapper for provider {provider_id}: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        db.close()


# ===== PATTERN 1: APSCHEDULER JOB =====
def probe_single_provider_job(provider_id: int):
    """
    APScheduler job for probing a single provider.

    Use this when you need to trigger individual provider probes
    from within the application (e.g., via API or scheduled triggers).
    """
    lock_key = f"probe_provider_{provider_id}"

    with redis_lock(lock_key, ttl=300):  # 5 minute TTL
        logger.info(f"Starting APScheduler job for provider {provider_id}")

        try:
            result = asyncio.run(probe_provider_minimal(provider_id))
            logger.info(
                f"APScheduler job completed for provider {provider_id}: {result['status']}"
            )
            return result
        except Exception as e:
            logger.error(f"APScheduler job failed for provider {provider_id}: {e}")
            raise


# ===== PATTERN 2: CRONJOB WRAPPER =====
def probe_provider_cronjob(provider_id: int):
    """
    Standalone wrapper for Kubernetes CronJob execution.

    Use this in a separate script/container for scheduled individual provider probes.
    Includes Redis locking to prevent overlapping runs.
    """
    import os
    import redis
    from contextlib import contextmanager

    # Redis configuration (same as probe_worker.py)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = redis.from_url(REDIS_URL)

    @contextmanager
    def redis_lock(lock_key: str, ttl: int = 300):
        got_lock = redis_client.set(name=lock_key, value="1", nx=True, ex=ttl)
        try:
            if not got_lock:
                logger.info(f"Lock {lock_key} already held")
                yield False
            else:
                logger.info(f"Acquired lock {lock_key}")
                yield True
        finally:
            if got_lock:
                redis_client.delete(lock_key)
                logger.info(f"Released lock {lock_key}")

    lock_key = f"provider_probe:{provider_id}"

    with redis_lock(lock_key, ttl=60) as acquired:  # 1 minute lock for single provider
        if not acquired:
            logger.info(f"Another probe for provider {provider_id} is running")
            return

        logger.info(f"Starting CronJob probe for provider {provider_id}")

        try:
            result = asyncio.run(probe_provider_minimal(provider_id))
            logger.info(
                f"CronJob probe completed for provider {provider_id}: {result['status']}"
            )
            return result
        except Exception as e:
            logger.error(f"CronJob probe failed for provider {provider_id}: {e}")
            raise


# ===== USAGE EXAMPLES =====


# Example 1: APScheduler integration
def register_single_provider_jobs(scheduler):
    """
    Register individual provider probe jobs with APScheduler.

    Call this during application startup to enable on-demand provider probing.
    """
    # This would be called from scheduler.py or main app
    # scheduler.add_job(
    #     probe_single_provider_job,
    #     args=[provider_id],
    #     id=f"probe_provider_{provider_id}",
    #     replace_existing=True
    # )
    pass


# Example 2: API endpoint integration
def create_api_endpoints(app):
    """
    Add API endpoints for manual provider probing.

    Example FastAPI/Pydantic integration.
    """
    from fastapi import HTTPException

    @app.post("/providers/{provider_id}/probe")
    async def probe_provider_endpoint(provider_id: int):
        """
        API endpoint to manually trigger provider probe.

        Use BackgroundTasks if this should run asynchronously:
        async def probe_provider_endpoint(provider_id: int, background_tasks: BackgroundTasks)
        """
        result = await probe_provider_minimal(provider_id)

        if result["status"] == "error":
            raise HTTPException(
                status_code=500, detail=result.get("error", "Probe failed")
            )

        return result


# Example 3: CronJob script
def main():
    """
    Main function for CronJob execution.

    Usage: python probe_single_provider.py <provider_id>
    """
    import sys

    if len(sys.argv) != 2:
        print("Usage: python probe_single_provider.py <provider_id>")
        sys.exit(1)

    try:
        provider_id = int(sys.argv[1])
    except ValueError:
        print("Error: provider_id must be an integer")
        sys.exit(1)

    # Run the CronJob wrapper
    result = probe_provider_cronjob(provider_id)
    print(f"Probe result: {result}")


if __name__ == "__main__":
    main()
