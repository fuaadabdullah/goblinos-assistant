"""
Bulkhead pattern implementation for limiting concurrent requests per provider.

Uses Redis-based semaphores to prevent resource exhaustion from too many
concurrent requests to the same provider.
"""

import redis
import logging
from typing import Optional, Callable
from contextlib import asynccontextmanager
import asyncio
import os
import threading

logger = logging.getLogger(__name__)


class BulkheadExceeded(Exception):
    """Exception raised when bulkhead limit is exceeded."""

    pass


class Bulkhead:
    """Redis-based bulkhead for limiting concurrent requests."""

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        redis_client: Optional[redis.Redis] = None,
    ):
        """Initialize bulkhead.

        Args:
            name: Unique name for this bulkhead
            max_concurrent: Maximum concurrent requests allowed
            redis_client: Redis client (will create if None)
        """
        self.name = name
        self.max_concurrent = max_concurrent
        self.counter_key = f"bulkhead:{name}:counter"
        self.redis_client = redis_client or self._create_redis_client()
        # Local in-memory fallback for single-process deployments or when Redis is unavailable.
        # This provides non-blocking "try acquire" semantics (fail fast) similar to the Redis path.
        self._local_lock = threading.Lock()
        self._local_count = 0

    def _create_redis_client(self) -> Optional[redis.Redis]:
        """Create Redis client from environment."""
        try:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                return redis.from_url(redis_url, decode_responses=True)
            else:
                host = os.getenv("REDIS_HOST", "localhost")
                port = int(os.getenv("REDIS_PORT", "6379"))
                db = int(os.getenv("REDIS_DB", "0"))
                password = os.getenv("REDIS_PASSWORD")
                ssl = os.getenv("REDIS_SSL", "false").lower() == "true"

                client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    ssl=ssl,
                    decode_responses=True,
                )
                client.ping()
                return client
        except Exception:
            logger.warning(f"Failed to connect to Redis for bulkhead: {self.name}")
            return None

    def acquire(self) -> bool:
        """Try to acquire a slot in the bulkhead.

        Returns:
            True if acquired, False if at limit
        """
        if not self.redis_client:
            with self._local_lock:
                if self._local_count >= self.max_concurrent:
                    return False
                self._local_count += 1
                return True

        # Use Lua script for atomic increment and check
        lua_script = """
        local counter_key = KEYS[1]
        local max_concurrent = tonumber(ARGV[1])

        local current = redis.call('GET', counter_key)
        if not current then
            current = 0
        else
            current = tonumber(current)
        end

        if current >= max_concurrent then
            return 0
        end

        redis.call('INCR', counter_key)
        return 1
        """

        try:
            result = self.redis_client.eval(
                lua_script, 1, self.counter_key, self.max_concurrent
            )
            return result == 1
        except Exception as e:
            logger.warning(f"Failed to acquire bulkhead slot: {e}")
            return True  # Allow on error

    def release(self):
        """Release a slot in the bulkhead."""
        if not self.redis_client:
            with self._local_lock:
                if self._local_count > 0:
                    self._local_count -= 1
            return

        try:
            self.redis_client.decr(self.counter_key)
        except Exception as e:
            logger.warning(f"Failed to release bulkhead slot: {e}")

    def get_status(self) -> dict:
        """Get bulkhead status."""
        status = {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "available_slots": self.max_concurrent,
            "current_concurrent": 0,
        }

        if self.redis_client:
            try:
                current = self.redis_client.get(self.counter_key)
                current_count = int(current) if current else 0
                status["current_concurrent"] = current_count
                status["available_slots"] = max(0, self.max_concurrent - current_count)
            except Exception as e:
                logger.warning(f"Failed to get bulkhead status: {e}")
        else:
            with self._local_lock:
                current_count = int(self._local_count)
            status["current_concurrent"] = current_count
            status["available_slots"] = max(0, self.max_concurrent - current_count)

        return status

    @asynccontextmanager
    async def guard(self):
        """Async context manager for bulkhead protection."""
        acquired = self.acquire()
        if not acquired:
            raise BulkheadExceeded(f"Bulkhead {self.name} limit exceeded")

        try:
            yield
        finally:
            self.release()

    def __call__(self, fn: Callable) -> Callable:
        """Decorator to wrap functions with bulkhead protection."""

        async def async_wrapper(*args, **kwargs):
            async with self.guard():
                return await fn(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            acquired = self.acquire()
            if not acquired:
                raise BulkheadExceeded(f"Bulkhead {self.name} limit exceeded")

            try:
                return fn(*args, **kwargs)
            finally:
                self.release()

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        else:
            return sync_wrapper


# Global bulkhead registry
_bulkheads = {}


def get_bulkhead(name: str, **kwargs) -> Bulkhead:
    """Get or create a bulkhead instance."""
    if name not in _bulkheads:
        _bulkheads[name] = Bulkhead(name, **kwargs)
    return _bulkheads[name]
