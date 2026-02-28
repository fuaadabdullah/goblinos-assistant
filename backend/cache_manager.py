#!/usr/bin/env python3
"""
Production Caching Layer for Goblin Assistant
Provides Redis-based caching with TTL, serialization, and error handling
"""

import asyncio
import json
import logging
import os
from typing import Any, Optional
import redis.asyncio as redis
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager with async support"""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis: Optional[redis.Redis] = None
        self._lock = asyncio.Lock()

    async def _get_redis(self) -> redis.Redis:
        """Lazy initialization of Redis connection"""
        if self._redis is None:
            async with self._lock:
                if self._redis is None:
                    try:
                        self._redis = redis.from_url(
                            self.redis_url,
                            encoding="utf-8",
                            decode_responses=True,
                            retry_on_timeout=True,
                            socket_connect_timeout=5,
                            socket_timeout=5,
                        )
                        # Test connection
                        await self._redis.ping()
                        logger.info("Redis cache connection established")
                    except Exception as e:
                        logger.error(f"Failed to connect to Redis: {e}")
                        self._redis = None
                        raise
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            redis_client = await self._get_redis()
            value = await redis_client.get(key)
            if value is not None:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
        return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL (seconds)"""
        try:
            redis_client = await self._get_redis()
            serialized = json.dumps(value)
            await redis_client.set(key, serialized, ex=ttl)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            redis_client = await self._get_redis()
            await redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis_client = await self._get_redis()
            return bool(await redis_client.exists(key))
        except Exception as e:
            logger.warning(f"Cache exists check failed for key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        try:
            redis_client = await self._get_redis()
            keys = await redis_client.keys(pattern)
            if keys:
                await redis_client.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.warning(f"Cache clear pattern failed for {pattern}: {e}")
            return 0

    async def get_ttl(self, key: str) -> int:
        """Get TTL for key in seconds"""
        try:
            redis_client = await self._get_redis()
            return await redis_client.ttl(key)
        except Exception as e:
            logger.warning(f"Cache TTL check failed for key {key}: {e}")
            return -1

    @asynccontextmanager
    async def pipeline(self):
        """Context manager for Redis pipeline operations"""
        redis_client = await self._get_redis()
        async with redis_client.pipeline() as pipe:
            yield pipe

    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._redis = None


# Global cache instance
_cache_instance: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance


# Convenience functions for common operations
async def cache_get(key: str) -> Optional[Any]:
    """Convenience function to get from cache"""
    return await get_cache_manager().get(key)


async def cache_set(key: str, value: Any, ttl: int = 300) -> bool:
    """Convenience function to set in cache"""
    return await get_cache_manager().set(key, value, ttl)


async def cache_delete(key: str) -> bool:
    """Convenience function to delete from cache"""
    return await get_cache_manager().delete(key)


# Cache key patterns
class CacheKeys:
    """Standardized cache key patterns"""

    @staticmethod
    def conversation(user_id: str, session_id: str) -> str:
        return f"conv:{user_id}:{session_id}"

    @staticmethod
    def user_profile(user_id: str) -> str:
        return f"user:{user_id}:profile"

    @staticmethod
    def llm_response(prompt_hash: str, model: str) -> str:
        return f"llm:{model}:{prompt_hash}"

    @staticmethod
    def api_rate_limit(identifier: str) -> str:
        return f"ratelimit:{identifier}"

    @staticmethod
    def feature_flag(flag_name: str) -> str:
        return f"feature:{flag_name}"


# Example usage and cache-aside pattern
async def get_with_cache(key: str, fetch_func, ttl: int = 300):
    """
    Cache-aside pattern: try cache first, then fetch if miss

    Usage:
        data = await get_with_cache(
            key="user:123:profile",
            fetch_func=lambda: db.get_user_profile(123),
            ttl=600
        )
    """
    # Try cache first
    cached = await cache_get(key)
    if cached is not None:
        return cached

    # Cache miss - fetch from source
    try:
        data = await fetch_func()
        if data is not None:
            await cache_set(key, data, ttl)
        return data
    except Exception as e:
        logger.error(f"Failed to fetch data for cache key {key}: {e}")
        return None


if __name__ == "__main__":
    # Quick test
    async def test_cache():
        cache = get_cache_manager()

        # Test basic operations
        await cache.set("test:key", {"message": "hello world"}, ttl=60)
        result = await cache.get("test:key")
        print(f"Cache test result: {result}")

        await cache.close()

    asyncio.run(test_cache())