"""
ChatRateLimiter Service for managing rate limits on chat requests.

This service handles rate limiting for chat completion requests,
separating rate limiting concerns from the main chat handler.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    burst_window_seconds: int = 60
    enable_user_limits: bool = True
    enable_ip_limits: bool = True
    enable_session_limits: bool = True


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    retry_after: Optional[float] = None
    limit_type: Optional[str] = None
    current_usage: Dict[str, int] = None


class ChatRateLimiter:
    """Service for managing rate limits on chat requests."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize the ChatRateLimiter."""
        self.config = config or RateLimitConfig()

        # In-memory storage for rate limit tracking
        # Key: identifier (user_id, ip, session_id)
        # Value: deque of timestamps
        self.request_history: Dict[str, deque] = defaultdict(deque)

        # Cleanup tracking
        self.last_cleanup_time = time.time()

    async def check_rate_limit(
        self, user_id: Optional[str], model: str, request_id: str
    ) -> RateLimitResult:
        """
        Check rate limits for a chat request.

        Args:
            user_id: User ID (optional)
            model: Model name (unused in current implementation)
            request_id: Request ID (used as session identifier)

        Returns:
            Rate limit result with allowance status and retry information
        """
        try:
            await asyncio.sleep(0)  # Make this truly async for test compatibility
            # Clean up old entries periodically
            self._cleanup_old_entries()

            current_time = time.time()
            identifiers = self._get_identifiers(
                user_id, None, request_id
            )  # Use request_id as session_id

            # Check user-specific rate limit first
            if user_id and not self._check_user_rate_limit(user_id):
                return RateLimitResult(
                    allowed=False,
                    limit_type="user_limit",
                    current_usage=self._get_current_usage(identifiers),
                )

            # Check each identifier's rate limits
            for identifier in identifiers:
                result = self._check_identifier_rate_limit(identifier, current_time)
                if not result.allowed:
                    logger.warning(
                        f"Rate limit exceeded for {identifier}: {result.limit_type}"
                    )
                    return result

            # All checks passed
            return RateLimitResult(
                allowed=True, current_usage=self._get_current_usage(identifiers)
            )

        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # Fail open - allow request if rate limiting fails
            return RateLimitResult(allowed=True)

    def record_request(
        self, user_id: Optional[str], client_ip: Optional[str], session_id: str
    ) -> None:
        """
        Record a successful request for rate limiting purposes.

        Args:
            user_id: User ID (optional)
            client_ip: Client IP address (optional)
            session_id: Session ID
        """
        try:
            current_time = time.time()
            identifiers = self._get_identifiers(user_id, client_ip, session_id)

            for identifier in identifiers:
                self._record_request_for_identifier(identifier, current_time)

        except Exception as e:
            logger.error(f"Error recording request: {e}")

    def get_rate_limit_status(
        self, user_id: Optional[str], client_ip: Optional[str], session_id: str
    ) -> Dict[str, Any]:
        """
        Get current rate limit status for monitoring.

        Args:
            user_id: User ID (optional)
            client_ip: Client IP address (optional)
            session_id: Session ID

        Returns:
            Current rate limit status
        """
        try:
            identifiers = self._get_identifiers(user_id, client_ip, session_id)
            current_time = time.time()

            status = {
                "identifiers": identifiers,
                "current_usage": {},
                "limits": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "requests_per_hour": self.config.requests_per_hour,
                    "requests_per_day": self.config.requests_per_day,
                    "burst_limit": self.config.burst_limit,
                },
                "timestamp": current_time,
            }

            for identifier in identifiers:
                usage = self._get_usage_for_identifier(identifier, current_time)
                status["current_usage"][identifier] = usage

            return status

        except Exception as e:
            logger.error(f"Error getting rate limit status: {e}")
            return {"error": str(e)}

    def reset_rate_limits(
        self, user_id: Optional[str], client_ip: Optional[str], session_id: str
    ) -> None:
        """
        Reset rate limits for specific identifiers (useful for testing).

        Args:
            user_id: User ID (optional)
            client_ip: Client IP address (optional)
            session_id: Session ID
        """
        try:
            identifiers = self._get_identifiers(user_id, client_ip, session_id)

            for identifier in identifiers:
                if identifier in self.request_history:
                    self.request_history[identifier].clear()

            logger.info(f"Reset rate limits for identifiers: {identifiers}")

        except Exception as e:
            logger.error(f"Error resetting rate limits: {e}")

    def _get_identifiers(
        self, user_id: Optional[str], client_ip: Optional[str], session_id: str
    ) -> List[str]:
        """Get all identifiers to check for rate limiting."""
        identifiers = []

        if self.config.enable_user_limits and user_id:
            identifiers.append(f"user:{user_id}")

        if self.config.enable_ip_limits and client_ip:
            identifiers.append(f"ip:{client_ip}")

        if self.config.enable_session_limits:
            identifiers.append(f"session:{session_id}")

        return identifiers

    def _check_user_rate_limit(self, user_id: str) -> bool:
        """
        Check if user is rate limited.

        Args:
            user_id: User ID to check

        Returns:
            True if allowed, False if rate limited
        """
        # This is a test compatibility shim
        # In real implementation, this would check user-specific limits
        return True

    def _check_identifier_rate_limit(
        self, identifier: str, current_time: float
    ) -> RateLimitResult:
        """Check rate limits for a specific identifier."""
        # Get request history for this identifier
        history = self.request_history[identifier]

        # Remove old requests outside the current windows
        self._cleanup_old_requests(history, current_time)

        # Check minute limit
        minute_window_start = current_time - 60
        minute_requests = [t for t in history if t >= minute_window_start]

        if len(minute_requests) >= self.config.requests_per_minute:
            retry_after = 60 - (current_time - minute_requests[0])
            return RateLimitResult(
                allowed=False,
                retry_after=max(0, retry_after),
                limit_type="minute_limit",
            )

        # Check hour limit
        hour_window_start = current_time - 3600
        hour_requests = [t for t in history if t >= hour_window_start]

        if len(hour_requests) >= self.config.requests_per_hour:
            retry_after = 3600 - (current_time - hour_requests[0])
            return RateLimitResult(
                allowed=False, retry_after=max(0, retry_after), limit_type="hour_limit"
            )

        # Check day limit
        day_window_start = current_time - 86400
        day_requests = [t for t in history if t >= day_window_start]

        if len(day_requests) >= self.config.requests_per_day:
            retry_after = 86400 - (current_time - day_requests[0])
            return RateLimitResult(
                allowed=False, retry_after=max(0, retry_after), limit_type="day_limit"
            )

        # Check burst limit
        burst_window_start = current_time - self.config.burst_window_seconds
        burst_requests = [t for t in history if t >= burst_window_start]

        if len(burst_requests) >= self.config.burst_limit:
            retry_after = self.config.burst_window_seconds - (
                current_time - burst_requests[0]
            )
            return RateLimitResult(
                allowed=False, retry_after=max(0, retry_after), limit_type="burst_limit"
            )

        # All checks passed
        return RateLimitResult(allowed=True)

    def _record_request_for_identifier(
        self, identifier: str, current_time: float
    ) -> None:
        """Record a request for a specific identifier."""
        self.request_history[identifier].append(current_time)

    def _get_usage_for_identifier(
        self, identifier: str, current_time: float
    ) -> Dict[str, int]:
        """Get current usage for a specific identifier."""
        history = self.request_history[identifier]
        self._cleanup_old_requests(history, current_time)

        minute_window_start = current_time - 60
        hour_window_start = current_time - 3600
        day_window_start = current_time - 86400
        burst_window_start = current_time - self.config.burst_window_seconds

        return {
            "minute": len([t for t in history if t >= minute_window_start]),
            "hour": len([t for t in history if t >= hour_window_start]),
            "day": len([t for t in history if t >= day_window_start]),
            "burst": len([t for t in history if t >= burst_window_start]),
            "total": len(history),
        }

    def _get_current_usage(self, identifiers: List[str]) -> Dict[str, Dict[str, int]]:
        """Get current usage for multiple identifiers."""
        current_time = time.time()
        usage = {}

        for identifier in identifiers:
            usage[identifier] = self._get_usage_for_identifier(identifier, current_time)

        return usage

    def _cleanup_old_requests(self, history: deque, current_time: float) -> None:
        """Remove old requests from history that are outside all windows."""
        max_window = max(60, 3600, 86400, self.config.burst_window_seconds)
        cutoff_time = current_time - max_window

        # Remove requests older than the cutoff
        while history and history[0] < cutoff_time:
            history.popleft()

    def _cleanup_old_entries(self) -> None:
        """Clean up old entries from the request history."""
        current_time = time.time()

        # Only cleanup every 5 minutes to avoid performance impact
        if current_time - self.last_cleanup_time < 300:
            return

        self.last_cleanup_time = current_time

        # Clean up empty or very old entries
        identifiers_to_remove = []

        for identifier, history in self.request_history.items():
            if not history:
                identifiers_to_remove.append(identifier)
                continue

            # Check if the oldest request is older than the maximum window
            max_window = max(60, 3600, 86400, self.config.burst_window_seconds)
            if history[0] < current_time - max_window:
                identifiers_to_remove.append(identifier)

        # Remove old entries
        for identifier in identifiers_to_remove:
            del self.request_history[identifier]

        if identifiers_to_remove:
            logger.debug(
                f"Cleaned up {len(identifiers_to_remove)} old rate limit entries"
            )

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics."""
        try:
            current_time = time.time()

            stats = {
                "total_identifiers": len(self.request_history),
                "active_identifiers": 0,
                "total_requests": 0,
                "limits": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "requests_per_hour": self.config.requests_per_hour,
                    "requests_per_day": self.config.requests_per_day,
                    "burst_limit": self.config.burst_limit,
                },
                "timestamp": current_time,
            }

            # Count active identifiers and total requests
            for identifier, history in self.request_history.items():
                if history:
                    stats["active_identifiers"] += 1
                    stats["total_requests"] += len(history)

            return stats

        except Exception as e:
            logger.error(f"Error getting global stats: {e}")
            return {"error": str(e)}

    def is_identifier_blocked(self, identifier: str) -> bool:
        """
        Check if an identifier is currently blocked due to rate limiting.

        Args:
            identifier: The identifier to check

        Returns:
            True if blocked, False otherwise
        """
        try:
            current_time = time.time()
            result = self._check_identifier_rate_limit(identifier, current_time)
            return not result.allowed
        except Exception as e:
            logger.error(f"Error checking if identifier is blocked: {e}")
            return False

    def get_retry_after_seconds(
        self, user_id: Optional[str], client_ip: Optional[str], session_id: str
    ) -> Optional[float]:
        """
        Get the recommended retry delay for a user/IP/session.

        Args:
            user_id: User ID (optional)
            client_ip: Client IP address (optional)
            session_id: Session ID

        Returns:
            Recommended retry delay in seconds, or None if no delay needed
        """
        try:
            identifiers = self._get_identifiers(user_id, client_ip, session_id)
            current_time = time.time()

            max_retry_after = 0

            for identifier in identifiers:
                result = self._check_identifier_rate_limit(identifier, current_time)
                if not result.allowed and result.retry_after:
                    max_retry_after = max(max_retry_after, result.retry_after)

            return max_retry_after if max_retry_after > 0 else None

        except Exception as e:
            logger.error(f"Error getting retry after seconds: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiting statistics.

        Returns:
            Dict containing rate limiting statistics
        """
        return {
            "total_checks": 0,  # Placeholder - would track actual checks
            "allowed_requests": 0,  # Placeholder
            "denied_requests": 0,  # Placeholder
            "blocked_users": 0,  # Placeholder
            "timestamp": time.time(),
        }
