"""
Challenge store for passkey challenges and temporary values.

Provides an async API with two concrete implementations:
- Redis-backed (if USE_REDIS_CHALLENGES=true and REDIS_URL configured)
- In-memory fallback (for local dev & tests)

Public factory: get_challenge_store_instance()
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Dict, Optional

try:
    # redis-py 4.x exposes asyncio support under redis.asyncio
    import redis.asyncio as aioredis
except Exception:  # pragma: no cover - redis optional
    aioredis = None


class BaseChallengeStore:
    async def set_challenge(
        self,
        email: str,
        challenge: str,
        expires_seconds: Optional[int] = None,
        ttl_minutes: Optional[int] = None,
    ) -> None:
        raise NotImplementedError()

    async def get_challenge(self, email: str) -> Optional[str]:
        raise NotImplementedError()

    async def delete_challenge(self, email: str) -> bool:
        raise NotImplementedError()

    async def cleanup_expired(self) -> int:
        """Return number of cleaned entries."""
        raise NotImplementedError()

    async def health_check(self) -> Dict[str, Any]:
        """Check store health."""
        raise NotImplementedError()


class InMemoryChallengeStore(BaseChallengeStore):
    def __init__(self):
        # store: email -> (challenge, expires_at)
        self._store: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()

    async def set_challenge(
        self,
        email: str,
        challenge: str,
        expires_seconds: Optional[int] = None,
        ttl_minutes: Optional[int] = None,
    ) -> None:
        if expires_seconds is None:
            expires_seconds = (ttl_minutes or 0) * 60
        expires_at = time.time() + int(expires_seconds)
        async with self._lock:
            self._store[email] = {"challenge": challenge, "expires_at": expires_at}

    async def get_challenge(self, email: str) -> Optional[str]:
        async with self._lock:
            entry = self._store.get(email)
            if not entry:
                return None
            if entry["expires_at"] < time.time():
                # expired
                del self._store[email]
                return None
            return entry["challenge"]

    async def delete_challenge(self, email: str) -> bool:
        async with self._lock:
            if email in self._store:
                del self._store[email]
                return True
            return False

    async def cleanup_expired(self) -> int:
        now = time.time()
        removed = 0
        async with self._lock:
            keys = list(self._store.keys())
            for k in keys:
                entry = self._store.get(k)
                if entry and entry.get("expires_at", 0) < now:
                    del self._store[k]
                    removed += 1
        return removed

    async def health_check(self) -> Dict[str, Any]:
        return {
            "store_type": "in_memory",
            "active_challenges": len(self._store),
            "healthy": True,
        }


class RedisChallengeStore(BaseChallengeStore):
    def __init__(self, url: str = "redis://localhost:6379/0"):
        if aioredis is None:
            raise RuntimeError("redis.asyncio is not available")
        self._client = aioredis.from_url(url, decode_responses=True)

    async def set_challenge(
        self,
        email: str,
        challenge: str,
        expires_seconds: Optional[int] = None,
        ttl_minutes: Optional[int] = None,
    ) -> None:
        if not self._client:
            raise RuntimeError("redis client not available")
        if expires_seconds is None:
            expires_seconds = (ttl_minutes or 0) * 60
        await self._client.set(email, challenge, ex=int(expires_seconds))

    async def get_challenge(self, email: str) -> Optional[str]:
        if not self._client:
            return None
        return await self._client.get(email)

    async def delete_challenge(self, email: str) -> bool:
        if not self._client:
            return False
        deleted_count = await self._client.delete(email)
        return deleted_count > 0

    async def cleanup_expired(self) -> int:
        # Redis expiry is automatic; return 0 to indicate no local cleanup
        return 0

    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            if not self._client:
                return {"redis_available": False, "error": "No client"}

            # Call info asynchronously
            info = {}
            if hasattr(self._client, "info"):
                res = self._client.info(section="all")
                if asyncio.iscoroutine(res):
                    info = await res
                else:
                    info = res

            return {
                "redis_available": True,
                "memory_used": info.get("used_memory_human", "unknown"),
                "memory_peak": info.get("used_memory_peak_human", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            return {"redis_available": False, "error": str(e)}


_singleton_store: Optional[BaseChallengeStore] = None


def get_challenge_store_instance() -> BaseChallengeStore:
    """Factory for a challenge store instance.

    Uses the environment variable USE_REDIS_CHALLENGES to decide whether to
    create a Redis-backed store. Falls back to an in-memory store.
    Returns the singleton instance.
    """
    global _singleton_store
    if _singleton_store is not None:
        return _singleton_store

    use_redis = os.getenv("USE_REDIS_CHALLENGES", "false").lower() == "true"
    if use_redis and aioredis:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            _singleton_store = RedisChallengeStore(redis_url)
            return _singleton_store
        except Exception:  # pragma: no cover - runtime-dependent
            # Fall back to in-memory if redis init fails
            _singleton_store = InMemoryChallengeStore()
            return _singleton_store

    _singleton_store = InMemoryChallengeStore()
    return _singleton_store
