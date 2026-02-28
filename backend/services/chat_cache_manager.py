"""
ChatCacheManager Service for managing chat response caching.

This service handles caching for chat completion responses,
separating caching concerns from the main chat handler.
"""

import logging
import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of cache entries."""

    RESPONSE = "response"
    EMBEDDING = "embedding"
    METADATA = "metadata"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    cache_type: CacheType
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    last_accessed: float = 0.0
    size: int = 0


class ChatCacheManager:
    """Service for managing chat response caching."""

    def __init__(
        self,
        default_ttl_seconds: int = 3600,  # 1 hour
        max_cache_size_mb: int = 100,  # 100 MB
        cleanup_interval_seconds: int = 300,  # 5 minutes
    ):
        """Initialize the ChatCacheManager."""
        self.default_ttl_seconds = default_ttl_seconds
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.cleanup_interval_seconds = cleanup_interval_seconds

        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}

        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size": 0,
            "entries": 0,
        }

        # Cleanup tracking
        self.last_cleanup_time = time.time()

    def get(
        self, key: str, cache_type: CacheType = CacheType.RESPONSE
    ) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key
            cache_type: Type of cache entry

        Returns:
            Cached value, or None if not found or expired
        """
        try:
            current_time = time.time()

            # Clean up expired entries periodically
            self._cleanup_expired_entries()

            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            entry = self.cache[key]

            # Check if entry has expired
            if entry.expires_at and current_time > entry.expires_at:
                self._remove_entry(key)
                self.stats["misses"] += 1
                return None

            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = current_time
            self.stats["hits"] += 1

            logger.debug(f"Cache hit for key: {key}")
            return entry.value

        except Exception as e:
            logger.error(f"Error getting cache entry for key {key}: {e}")
            self.stats["misses"] += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        cache_type: CacheType = CacheType.RESPONSE,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache entry
            ttl_seconds: Time to live in seconds (optional)

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            current_time = time.time()
            ttl = ttl_seconds or self.default_ttl_seconds
            expires_at = current_time + ttl

            # Serialize value to estimate size
            serialized_value = json.dumps(value, default=str)
            size = len(serialized_value.encode("utf-8"))

            # Check if value is too large
            if size > self.max_cache_size_bytes:
                logger.warning(f"Value too large to cache: {size} bytes")
                return False

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                cache_type=cache_type,
                created_at=current_time,
                expires_at=expires_at,
                access_count=0,
                last_accessed=current_time,
                size=size,
            )

            # Check if we need to make space
            self._ensure_space(entry.size)

            # Add to cache
            self.cache[key] = entry
            self.stats["entries"] += 1
            self.stats["total_size"] += size

            logger.debug(f"Cached value for key: {key}, size: {size} bytes")
            return True

        except Exception as e:
            logger.error(f"Error setting cache entry for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            True if successfully deleted, False if not found
        """
        try:
            if key in self.cache:
                self._remove_entry(key)
                logger.debug(f"Deleted cache entry for key: {key}")
                return True
            else:
                logger.debug(f"Cache entry not found for key: {key}")
                return False

        except Exception as e:
            logger.error(f"Error deleting cache entry for key {key}: {e}")
            return False

    def clear(self, cache_type: Optional[CacheType] = None) -> int:
        """
        Clear cache entries.

        Args:
            cache_type: Specific cache type to clear (optional)

        Returns:
            Number of entries cleared
        """
        try:
            if cache_type:
                # Clear specific cache type
                keys_to_remove = [
                    key
                    for key, entry in self.cache.items()
                    if entry.cache_type == cache_type
                ]
            else:
                # Clear all cache
                keys_to_remove = list(self.cache.keys())

            for key in keys_to_remove:
                self._remove_entry(key)

            logger.info(f"Cleared {len(keys_to_remove)} cache entries")
            return len(keys_to_remove)

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            current_time = time.time()

            # Calculate hit rate
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / max(total_requests, 1)

            # Get cache age statistics
            if self.cache:
                ages = [
                    current_time - entry.created_at for entry in self.cache.values()
                ]
                avg_age = sum(ages) / len(ages)
                max_age = max(ages)
                min_age = min(ages)
            else:
                avg_age = max_age = min_age = 0.0

            return {
                "entries": self.stats["entries"],
                "total_size": self.stats["total_size"],
                "max_size": self.max_cache_size_bytes,
                "hit_rate": hit_rate,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "avg_age_seconds": avg_age,
                "max_age_seconds": max_age,
                "min_age_seconds": min_age,
                "cache_types": self._get_cache_type_stats(),
                "timestamp": current_time,
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    def generate_key(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        cache_type: CacheType = CacheType.RESPONSE,
    ) -> str:
        """
        Generate a cache key for chat requests.

        Args:
            session_id: Session ID
            messages: List of chat messages
            model: Model name
            temperature: Temperature setting
            max_tokens: Maximum tokens
            cache_type: Type of cache entry

        Returns:
            Generated cache key
        """
        try:
            # Create a hashable representation of the request
            request_data = {
                "session_id": session_id,
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "cache_type": cache_type.value,
            }

            # Serialize and hash
            serialized = json.dumps(request_data, sort_keys=True)
            key_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

            return f"{cache_type.value}:{key_hash}"

        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            # Fallback to a simple key
            return f"{cache_type.value}:fallback_{hash(str(messages))}"

    def cleanup_expired_entries(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of entries cleaned up
        """
        try:
            current_time = time.time()
            expired_keys = [
                key
                for key, entry in self.cache.items()
                if entry.expires_at and current_time > entry.expires_at
            ]

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
            return 0

    def get_cache_health(self) -> Dict[str, Any]:
        """Get cache health information."""
        try:
            stats = self.get_stats()
            current_time = time.time()

            # Check for potential issues
            issues = []

            # Check cache size
            if stats["total_size"] > self.max_cache_size_bytes * 0.9:
                issues.append("Cache size approaching limit")

            # Check hit rate
            if stats["hit_rate"] < 0.1:
                issues.append("Low cache hit rate")

            # Check entry ages
            if stats["avg_age_seconds"] > self.default_ttl_seconds * 0.8:
                issues.append("Many old cache entries")

            return {
                "status": "healthy" if not issues else "warning",
                "issues": issues,
                "stats": stats,
                "timestamp": current_time,
            }

        except Exception as e:
            logger.error(f"Error getting cache health: {e}")
            return {"status": "error", "error": str(e)}

    def _remove_entry(self, key: str) -> None:
        """Remove a cache entry and update statistics."""
        if key in self.cache:
            entry = self.cache[key]
            del self.cache[key]
            self.stats["entries"] -= 1
            self.stats["total_size"] -= entry.size

    def _ensure_space(self, new_entry_size: int) -> None:
        """Ensure there's enough space for a new entry by evicting old ones."""
        current_time = time.time()

        # If we have enough space, return early
        if self.stats["total_size"] + new_entry_size <= self.max_cache_size_bytes:
            return

        # Sort entries by last accessed time
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].last_accessed)

        # Evict entries until we have enough space
        evicted_count = 0
        for key, entry in sorted_entries:
            if self.stats["total_size"] + new_entry_size <= self.max_cache_size_bytes:
                break

            self._remove_entry(key)
            self.stats["evictions"] += 1
            evicted_count += 1

        if evicted_count > 0:
            logger.info(f"Evicted {evicted_count} cache entries to make space")

    def _cleanup_expired_entries(self) -> None:
        """Clean up expired entries if enough time has passed."""
        current_time = time.time()

        # Only cleanup every cleanup_interval_seconds
        if current_time - self.last_cleanup_time < self.cleanup_interval_seconds:
            return

        self.last_cleanup_time = current_time
        self.cleanup_expired_entries()

    def _get_cache_type_stats(self) -> Dict[str, int]:
        """Get statistics by cache type."""
        type_counts = {}
        for entry in self.cache.values():
            cache_type = entry.cache_type.value
            type_counts[cache_type] = type_counts.get(cache_type, 0) + 1
        return type_counts

    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific cache entry.

        Args:
            key: Cache key

        Returns:
            Entry information, or None if not found
        """
        try:
            if key not in self.cache:
                return None

            entry = self.cache[key]
            current_time = time.time()

            return {
                "key": entry.key,
                "cache_type": entry.cache_type.value,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
                "size": entry.size,
                "age_seconds": current_time - entry.created_at,
                "time_until_expiry": (entry.expires_at - current_time)
                if entry.expires_at
                else None,
            }

        except Exception as e:
            logger.error(f"Error getting entry info for key {key}: {e}")
            return None

    def get_entries_by_type(self, cache_type: CacheType) -> List[str]:
        """
        Get all cache keys for a specific cache type.

        Args:
            cache_type: Type of cache entries to retrieve

        Returns:
            List of cache keys
        """
        try:
            return [
                key
                for key, entry in self.cache.items()
                if entry.cache_type == cache_type
            ]

        except Exception as e:
            logger.error(f"Error getting entries by type {cache_type}: {e}")
            return []

    def get_largest_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the largest cache entries by size.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of largest entries with their info
        """
        try:
            sorted_entries = sorted(
                self.cache.items(), key=lambda x: x[1].size, reverse=True
            )[:limit]

            return [
                {
                    "key": entry.key,
                    "size": entry.size,
                    "cache_type": entry.cache_type.value,
                    "access_count": entry.access_count,
                }
                for _, entry in sorted_entries
            ]

        except Exception as e:
            logger.error(f"Error getting largest entries: {e}")
            return []

    def get_most_accessed_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most frequently accessed cache entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of most accessed entries with their info
        """
        try:
            sorted_entries = sorted(
                self.cache.items(), key=lambda x: x[1].access_count, reverse=True
            )[:limit]

            return [
                {
                    "key": entry.key,
                    "access_count": entry.access_count,
                    "size": entry.size,
                    "cache_type": entry.cache_type.value,
                }
                for _, entry in sorted_entries
            ]

        except Exception as e:
            logger.error(f"Error getting most accessed entries: {e}")
            return []

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        try:
            self.stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "total_size": 0,
                "entries": 0,
            }
            logger.info("Reset cache statistics")

        except Exception as e:
            logger.error(f"Error resetting cache stats: {e}")

    def warm_cache(
        self, entries: List[Tuple[str, Any, CacheType, Optional[int]]]
    ) -> int:
        """
        Warm the cache with pre-loaded entries.

        Args:
            entries: List of (key, value, cache_type, ttl_seconds) tuples

        Returns:
            Number of entries successfully cached
        """
        try:
            success_count = 0
            for key, value, cache_type, ttl in entries:
                if self.set(key, value, cache_type, ttl):
                    success_count += 1

            logger.info(
                f"Warm cache: {success_count}/{len(entries)} entries cached successfully"
            )
            return success_count

        except Exception as e:
            logger.error(f"Error warming cache: {e}")
            return 0

    async def cleanup_session_cache(self, session_id: str) -> None:
        """
        Clean up cache entries for a specific session.

        Args:
            session_id: The session ID to clean up cache for
        """
        import asyncio

        try:
            # Remove all cache entries that contain the session_id
            keys_to_remove = []
            for key, entry in self.cache.items():
                if session_id in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.cache[key]
                self.stats["entries"] -= 1
                self.current_size -= getattr(entry, "size", 0)

            if keys_to_remove:
                logger.info(
                    f"Cleaned up {len(keys_to_remove)} cache entries for session {session_id}"
                )

            await asyncio.sleep(0)  # Make this properly async

        except Exception as e:
            logger.error(f"Error cleaning up session cache for {session_id}: {e}")
