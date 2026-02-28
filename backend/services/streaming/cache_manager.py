"""Cache manager for streaming responses."""

import time
from typing import Any


class StreamCacheManager:
    """Manager for caching streaming responses."""

    def __init__(self, cache_ttl: int = 300):
        """Initialize cache manager."""
        self.cache_ttl = cache_ttl
        self.cache: dict[str, tuple[Any, float]] = {}

    async def get_cached_response(self, cache_key: str) -> Any | None:
        """Get cached response if not expired."""
        if cache_key in self.cache:
            response, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return response
        return None

    async def cache_response(self, cache_key: str, response: Any) -> None:
        """Cache a response."""
        self.cache[cache_key] = (response, time.time())

    async def cleanup_session_cache(self, session_id: str) -> None:
        """Clean up cache for a session."""
        keys_to_remove = [key for key in self.cache.keys() if key.startswith(session_id)]
        for key in keys_to_remove:
            del self.cache[key]
