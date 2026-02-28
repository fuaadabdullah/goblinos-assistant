"""
Routing cache for decision caching and rate-windowing.

Provides sliding window caches to replace ad-hoc Redis spike detection.
"""

import time
import threading
from typing import Dict, Any, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class SlidingWindow:
    """Sliding window for tracking events over time."""

    def __init__(self, window_size: int = 60):
        """Initialize sliding window.

        Args:
            window_size: Window size in seconds
        """
        self.window_size = window_size
        self.events: deque = deque()
        self.lock = threading.Lock()

    def add_event(self, value: Any = 1, timestamp: Optional[float] = None):
        """Add an event to the window.

        Args:
            value: Event value (default 1 for counting)
            timestamp: Event timestamp (default current time)
        """
        if timestamp is None:
            timestamp = time.time()

        with self.lock:
            self.events.append((timestamp, value))
            self._cleanup()

    def get_sum(self, since: Optional[float] = None) -> float:
        """Get sum of events in window.

        Args:
            since: Only count events since this timestamp

        Returns:
            Sum of event values
        """
        cutoff = since or (time.time() - self.window_size)

        with self.lock:
            self._cleanup()
            return sum(value for ts, value in self.events if ts >= cutoff)

    def get_count(self, since: Optional[float] = None) -> int:
        """Get count of events in window.

        Args:
            since: Only count events since this timestamp

        Returns:
            Number of events
        """
        cutoff = since or (time.time() - self.window_size)

        with self.lock:
            self._cleanup()
            return sum(1 for ts, _ in self.events if ts >= cutoff)

    def get_rate(self, since: Optional[float] = None) -> float:
        """Get event rate (events per second).

        Args:
            since: Calculate rate since this timestamp

        Returns:
            Events per second
        """
        cutoff = since or (time.time() - self.window_size)
        now = time.time()
        window_duration = now - cutoff

        if window_duration <= 0:
            return 0.0

        count = self.get_count(since=cutoff)
        return count / window_duration

    def _cleanup(self):
        """Remove events outside the window."""
        cutoff = time.time() - self.window_size
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()


class RoutingCache:
    """Cache for routing decisions and rate limiting."""

    def __init__(self):
        """Initialize routing cache."""
        # Rate limiting windows (requests per minute per client)
        self.client_windows: Dict[str, SlidingWindow] = {}

        # Provider performance windows
        self.provider_latency: Dict[str, SlidingWindow] = {}
        self.provider_errors: Dict[str, SlidingWindow] = {}
        self.provider_requests: Dict[str, SlidingWindow] = {}

        # Decision cache (provider -> request_hash -> decision)
        self.decision_cache: Dict[str, Dict[str, Tuple[Any, float]]] = {}

        # Cache TTL in seconds
        self.cache_ttl = 300  # 5 minutes

        self.lock = threading.Lock()

    def check_rate_limit(
        self, client_key: str, limit: int, window: int = 60
    ) -> Tuple[bool, float]:
        """Check if client is within rate limit.

        Args:
            client_key: Client identifier
            limit: Maximum requests allowed
            window: Time window in seconds

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        if client_key not in self.client_windows:
            self.client_windows[client_key] = SlidingWindow(window)

        window_obj = self.client_windows[client_key]
        current_count = window_obj.get_count()

        if current_count >= limit:
            # Calculate retry after time
            if window_obj.events:
                oldest_timestamp = window_obj.events[0][0]
                retry_after = window - (time.time() - oldest_timestamp)
                retry_after = max(0, retry_after)
            else:
                retry_after = window
            return False, retry_after

        # Add this request to the window
        window_obj.add_event()
        return True, 0.0

    def record_provider_request(self, provider_id: str, latency_ms: int, success: bool):
        """Record provider request metrics.

        Args:
            provider_id: Provider identifier
            latency_ms: Request latency in milliseconds
            success: Whether request was successful
        """
        # Initialize windows if needed
        if provider_id not in self.provider_requests:
            self.provider_requests[provider_id] = SlidingWindow(3600)  # 1 hour
        if provider_id not in self.provider_latency:
            self.provider_latency[provider_id] = SlidingWindow(3600)
        if provider_id not in self.provider_errors:
            self.provider_errors[provider_id] = SlidingWindow(3600)

        # Record metrics
        self.provider_requests[provider_id].add_event()
        self.provider_latency[provider_id].add_event(latency_ms)

        if not success:
            self.provider_errors[provider_id].add_event()

    def get_provider_metrics(self, provider_id: str) -> Dict[str, Any]:
        """Get provider performance metrics.

        Args:
            provider_id: Provider identifier

        Returns:
            Dictionary with provider metrics
        """
        if provider_id not in self.provider_requests:
            return {
                "requests_per_hour": 0,
                "avg_latency_ms": 0,
                "error_rate": 0,
                "total_requests": 0,
            }

        requests_window = self.provider_requests[provider_id]
        latency_window = self.provider_latency.get(provider_id)
        errors_window = self.provider_errors.get(provider_id)

        total_requests = requests_window.get_count()
        requests_per_hour = requests_window.get_rate() * 3600

        avg_latency = 0
        if latency_window:
            # Calculate average latency from sliding window
            events = list(latency_window.events)
            if events:
                avg_latency = sum(value for _, value in events) / len(events)

        error_rate = 0
        if errors_window and total_requests > 0:
            error_count = errors_window.get_count()
            error_rate = error_count / total_requests

        return {
            "requests_per_hour": requests_per_hour,
            "avg_latency_ms": avg_latency,
            "error_rate": error_rate,
            "total_requests": total_requests,
        }

    def cache_decision(self, provider_id: str, request_hash: str, decision: Any):
        """Cache a routing decision.

        Args:
            provider_id: Provider identifier
            request_hash: Hash of the request context
            decision: Routing decision to cache
        """
        with self.lock:
            if provider_id not in self.decision_cache:
                self.decision_cache[provider_id] = {}

            self.decision_cache[provider_id][request_hash] = (decision, time.time())

    def get_cached_decision(self, provider_id: str, request_hash: str) -> Optional[Any]:
        """Get cached routing decision.

        Args:
            provider_id: Provider identifier
            request_hash: Hash of the request context

        Returns:
            Cached decision or None if not found/expired
        """
        with self.lock:
            if provider_id not in self.decision_cache:
                return None

            cache_entry = self.decision_cache[provider_id].get(request_hash)
            if not cache_entry:
                return None

            decision, timestamp = cache_entry
            if time.time() - timestamp > self.cache_ttl:
                # Expired, remove from cache
                del self.decision_cache[provider_id][request_hash]
                return None

            return decision

    def detect_spike(
        self, provider_id: str, threshold: float = 2.0, window: int = 60
    ) -> bool:
        """Detect traffic spikes for a provider.

        Args:
            provider_id: Provider identifier
            threshold: Spike threshold multiplier
            window: Detection window in seconds

        Returns:
            True if spike detected
        """
        if provider_id not in self.provider_requests:
            return False

        # Get current rate and baseline rate
        current_rate = self.provider_requests[provider_id].get_rate()

        # Use longer window for baseline
        baseline_window = window * 4
        baseline_rate = self._get_baseline_rate(provider_id, baseline_window)

        if baseline_rate == 0:
            return False

        return current_rate > (baseline_rate * threshold)

    def _get_baseline_rate(self, provider_id: str, window: int) -> float:
        """Get baseline rate for spike detection.

        Args:
            provider_id: Provider identifier
            window: Baseline window in seconds

        Returns:
            Baseline rate (requests per second)
        """
        if provider_id not in self.provider_requests:
            return 0

        # Get rate over longer baseline period
        baseline_start = time.time() - window
        return self.provider_requests[provider_id].get_rate(since=baseline_start)

    def cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        with self.lock:
            current_time = time.time()
            for provider_id in list(self.decision_cache.keys()):
                provider_cache = self.decision_cache[provider_id]
                expired_keys = [
                    key
                    for key, (_, timestamp) in provider_cache.items()
                    if current_time - timestamp > self.cache_ttl
                ]
                for key in expired_keys:
                    del provider_cache[key]

                if not provider_cache:
                    del self.decision_cache[provider_id]


# Global cache instance
_cache: Optional[RoutingCache] = None


def get_routing_cache() -> RoutingCache:
    """Get the global routing cache instance."""
    global _cache
    if _cache is None:
        _cache = RoutingCache()
    assert _cache is not None
    return _cache
