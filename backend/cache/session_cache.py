"""
Redis Session Cache for improved performance.

Provides a simple Redis-backed session cache to reduce database load
for frequently accessed session data.
"""

import json
import os
from typing import Optional, Any
import redis


class RedisSessionCache:
    """Redis-backed session cache for storing temporary session data."""

    def __init__(self):
        # Initialize Redis connection
        self.redis_client = self._init_redis()
        self.key_prefix = "session:"
        self.default_ttl = int(os.getenv("SESSION_CACHE_TTL", "1800"))  # 30 minutes

    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection."""
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
            return None

    def get(self, session_id: str, key: str) -> Optional[Any]:
        """Get a value from session cache."""
        if not self.redis_client:
            return None

        cache_key = f"{self.key_prefix}{session_id}:{key}"
        try:
            value = self.redis_client.get(cache_key)
            return json.loads(value) if value else None
        except Exception:
            return None

    def set(
        self, session_id: str, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set a value in session cache."""
        if not self.redis_client:
            return False

        cache_key = f"{self.key_prefix}{session_id}:{key}"
        try:
            json_value = json.dumps(value)
            return bool(
                self.redis_client.setex(cache_key, ttl or self.default_ttl, json_value)
            )
        except Exception:
            return False

    def delete(self, session_id: str, key: str) -> bool:
        """Delete a value from session cache."""
        if not self.redis_client:
            return False

        cache_key = f"{self.key_prefix}{session_id}:{key}"
        try:
            return bool(self.redis_client.delete(cache_key))
        except Exception:
            return False

    def clear_session(self, session_id: str) -> int:
        """Clear all cached data for a session."""
        if not self.redis_client:
            return 0

        try:
            # Use SCAN to find all keys for this session
            keys_to_delete = []
            cursor = 0
            pattern = f"{self.key_prefix}{session_id}:*"

            while True:
                cursor, keys = self.redis_client.scan(cursor, pattern, 100)
                keys_to_delete.extend(keys)
                if cursor == 0:
                    break

            if keys_to_delete:
                return self.redis_client.delete(*keys_to_delete)
            return 0
        except Exception:
            return 0

    def health_check(self) -> dict:
        """Check cache health."""
        if not self.redis_client:
            return {"available": False, "error": "Redis not configured"}

        try:
            self.redis_client.ping()
            info = self.redis_client.info("memory")
            return {
                "available": True,
                "memory_used": info.get("used_memory_human", "unknown"),
                "session_cache_ttl": self.default_ttl,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}


# Global session cache instance
_session_cache = None


def get_session_cache() -> RedisSessionCache:
    """Get singleton session cache instance."""
    global _session_cache
    if _session_cache is None:
        _session_cache = RedisSessionCache()
    return _session_cache
