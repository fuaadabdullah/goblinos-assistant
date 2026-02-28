"""Redis connection helpers for autoscaling."""

from contextlib import asynccontextmanager

import redis.asyncio as redis


async def initialize_redis(service) -> None:
    """Initialize Redis connection for autoscaling service."""
    if not service.redis:
        service.redis = redis.from_url(service.redis_url, decode_responses=True)
        await service.redis.ping()


async def close_redis(service) -> None:
    """Close Redis connection for autoscaling service."""
    if service.redis:
        await service.redis.close()
        service.redis = None


def get_redis_connection(service):
    """Return the active Redis connection, if any."""
    return service.redis


@asynccontextmanager
async def redis_connection(service):
    """Context manager for Redis operations."""
    if not service.redis:
        await initialize_redis(service)
    try:
        yield service.redis
    except Exception:
        raise
