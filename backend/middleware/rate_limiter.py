"""Configurable rate limiting middleware with Redis-backed storage for production.

Provides tiered rate limits configurable via environment variables:
- Auth endpoints: configurable (default: 10/minute)
- Chat endpoints: configurable (default: 30/minute)
- Health endpoints: configurable (default: 60/minute)
- General API: configurable (default: 100/minute)

Supports proper HTTP rate limit headers (RFC 6585):
- X-RateLimit-Limit: Maximum requests per window
- X-RateLimit-Remaining: Remaining requests in current window
- X-RateLimit-Reset: Time when limit resets (Unix timestamp)
- Retry-After: Seconds to wait before retrying (when limit exceeded)

Uses Redis for distributed rate limiting across multiple server instances.

Kong Integration:
- Checks for X-API-Gateway header to detect Kong-routed requests
- Uses Kong's rate limit headers when available
- Falls back to local rate limiting for direct requests
- Prevents double rate limiting
"""

import time
import os
import logging
from typing import Tuple
import redis
from fastapi import Request, HTTPException, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter:
    """Redis-backed rate limiter for production scaling."""

    def __init__(self):
        # Initialize Redis connection
        self.redis_client = self._init_redis()
        self.fallback_mode = self.redis_client is None

    def _init_redis(self) -> redis.Redis:
        """Initialize Redis connection with fallback to None."""
        try:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                return redis.from_url(redis_url)
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
                client.ping()  # Test connection
                return client
        except Exception:
            return None

    def is_allowed(
        self, client_ip: str, endpoint: str, limit: int, window: int = 60
    ) -> Tuple[bool, int, int, int]:
        """Check if request is allowed under rate limit using Redis.

        Args:
            client_ip: Client IP address
            endpoint: API endpoint pattern
            limit: Maximum requests allowed
            window: Time window in seconds (default: 60)

        Returns:
            Tuple of (is_allowed, retry_after_seconds, remaining_requests, reset_timestamp)
        """
        if self.fallback_mode or not self.redis_client:
            # Fallback to simple in-memory limiting if Redis unavailable
            return self._fallback_is_allowed(client_ip, endpoint, limit, window)

        now = int(time.time())
        key = f"ratelimit:{client_ip}:{endpoint}"

        try:
            # Use Redis sorted set to track requests within the window
            # Add current timestamp to the set
            self.redis_client.zadd(key, {str(now): now})

            # Remove requests outside the window
            window_start = now - window
            self.redis_client.zremrangebyscore(key, "-inf", window_start)

            # Count requests in current window
            current_count = self.redis_client.zcard(key)

            # Set expiration on the key (cleanup)
            self.redis_client.expire(key, window * 2)

            remaining = max(0, limit - current_count)
            reset_timestamp = now + window

            if current_count > limit:
                # Rate limit exceeded
                retry_after = window - (
                    now
                    - int(
                        self.redis_client.zrange(key, 0, 0)[0].decode()
                        if self.redis_client.zrange(key, 0, 0)
                        else now
                    )
                )
                return False, max(1, int(retry_after)), 0, reset_timestamp

            # Request allowed
            return True, 0, remaining - 1, reset_timestamp

        except Exception:
            # Redis error, fallback to in-memory
            return self._fallback_is_allowed(client_ip, endpoint, limit, window)

    def _fallback_is_allowed(
        self, client_ip: str, endpoint: str, limit: int, window: int = 60
    ) -> Tuple[bool, int, int, int]:
        """Simple in-memory fallback rate limiting."""
        # Initialize storage if not exists
        if not hasattr(self, "_fallback_requests"):
            from collections import defaultdict

            self._fallback_requests = defaultdict(lambda: defaultdict(list))

        now = time.time()
        window_start = now - window

        # Clean old requests
        self._fallback_requests[client_ip][endpoint] = [
            ts for ts in self._fallback_requests[client_ip][endpoint] if ts > window_start
        ]

        current_count = len(self._fallback_requests[client_ip][endpoint])
        remaining = max(0, limit - current_count)

        if current_count >= limit:
            if self._fallback_requests[client_ip][endpoint]:
                oldest = self._fallback_requests[client_ip][endpoint][0]
                retry_after = int(window - (now - oldest)) + 1
                reset_timestamp = int(oldest + window)
                return False, retry_after, 0, reset_timestamp
            reset_timestamp = int(now + window)
            return False, window, 0, reset_timestamp

        # Allow request
        self._fallback_requests[client_ip][endpoint].append(now)
        reset_timestamp = int(now + window)
        return True, 0, remaining - 1, reset_timestamp

    def health_check(self) -> dict:
        """
        Check Redis health and rate limiting status.

        Returns:
            dict: Health status including Redis metrics
        """
        if self.fallback_mode or not self.redis_client:
            return {
                "redis_available": False,
                "fallback_mode": True,
                "rate_limit_keys": len(getattr(self, "_fallback_requests", {})),
            }

        try:
            # Test basic connectivity
            self.redis_client.ping()

            # Get Redis info
            info = self.redis_client.info()
            memory_info = info.get("memory", {})

            # Count rate limit keys (approximate)
            rate_limit_keys = "unknown"
            try:
                # Use SCAN to count keys with ratelimit: prefix
                cursor = 0
                count = 0
                while True:
                    cursor, keys = self.redis_client.scan(cursor, "ratelimit:*", 1000)
                    count += len(keys)
                    if cursor == 0:
                        break
                rate_limit_keys = count
            except Exception:
                pass

            return {
                "redis_available": True,
                "fallback_mode": False,
                "memory_used": memory_info.get("used_memory_human", "unknown"),
                "memory_peak": memory_info.get("used_memory_peak_human", "unknown"),
                "connected_clients": info.get("clients", {}).get("connected_clients", "unknown"),
                "rate_limit_keys": rate_limit_keys,
                "uptime_seconds": info.get("server", {}).get("uptime_in_seconds", "unknown"),
            }
        except Exception as e:
            return {
                "redis_available": False,
                "fallback_mode": True,
                "error": str(e),
                "rate_limit_keys": len(getattr(self, "_fallback_requests", {})),
            }

    def cleanup_old_entries(self) -> int:
        """Best-effort cleanup for in-memory fallback storage.

        Redis-backed paths already expire keys, but the fallback in-memory
        limiter can grow without bound in long-running processes.
        """
        if not hasattr(self, "_fallback_requests"):
            return 0

        # Use configured window if present; keep a small safety buffer.
        try:
            window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        except Exception:
            window = 60
        cutoff = time.time() - (window * 2)

        removed = 0
        requests_by_ip = getattr(self, "_fallback_requests", {})
        for client_ip, endpoints in list(requests_by_ip.items()):
            for endpoint, timestamps in list(endpoints.items()):
                kept = [ts for ts in timestamps if ts > cutoff]
                if kept:
                    endpoints[endpoint] = kept
                else:
                    endpoints.pop(endpoint, None)
                    removed += 1

            if not endpoints:
                requests_by_ip.pop(client_ip, None)
                removed += 1

        return removed


# Redis-backed rate limiter (fixed window approach)
class RedisRateLimiter:
    """Redis-backed fixed window rate limiter based on simple INCR/EXPIRE counters."""

    def __init__(self, redis_url: str, timeout: int = 5):
        self.redis = redis.from_url(redis_url, socket_timeout=timeout)
        self._fallback_limiter = RateLimiter() if ALLOW_MEMORY_FALLBACK else None

    def is_allowed(self, client_ip: str, endpoint: str, limit: int, window: int = 60):
        now = int(time.time())
        bucket = now // window
        key = f"ratelimit:{client_ip}:{endpoint}:{bucket}"
        try:
            count = self.redis.incr(key)
            if count == 1:
                self.redis.expire(key, window)
        except Exception:
            # If Redis errors and fallback is allowed, use in-memory limiter.
            if self._fallback_limiter is not None:
                return self._fallback_limiter._fallback_is_allowed(
                    client_ip, endpoint, limit, window
                )
            raise

        allowed = count <= limit
        remaining = max(0, limit - count) if allowed else 0
        reset_timestamp = (bucket + 1) * window
        retry_after = 0 if allowed else int(reset_timestamp - now) + 1
        return allowed, retry_after, remaining, reset_timestamp

    def cleanup_old_entries(self) -> None:
        """No-op: keys are stored with TTLs so Redis handles cleanup."""
        return None


# Instantiate limiter based on REDIS_URL or fallback to in-memory
REDIS_URL = os.getenv("REDIS_URL") or os.getenv("REDIS_HOST")
if os.getenv("REDIS_HOST") and not REDIS_URL:
    REDIS_URL = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT', '6379')}/{os.getenv('REDIS_DB', '0')}"

# Development-friendly default: allow in-memory fallback unless explicitly disabled
# or running in production.
_env = (os.getenv("ENVIRONMENT") or os.getenv("environment") or "development").strip().lower()
_default_allow_fallback = "false" if _env == "production" else "true"
ALLOW_MEMORY_FALLBACK = (
    os.getenv("ALLOW_MEMORY_FALLBACK", _default_allow_fallback).lower() == "true"
)

if REDIS_URL:
    try:
        limiter = RedisRateLimiter(REDIS_URL)
    except Exception:
        if ALLOW_MEMORY_FALLBACK:
            limiter = RateLimiter()
        else:
            raise
else:
    if ALLOW_MEMORY_FALLBACK:
        limiter = RateLimiter()
    else:
        raise RuntimeError(
            "REDIS_URL not configured and memory fallback disallowed; configure Redis"
        )


# Load configurable rate limits from environment
RATE_LIMITS = {
    "/auth": int(os.getenv("RATE_LIMIT_AUTH", "10")),  # Default: 10 requests per minute
    "/chat": int(os.getenv("RATE_LIMIT_CHAT", "30")),  # Default: 30 requests per minute
    "/health": int(os.getenv("RATE_LIMIT_HEALTH", "60")),  # Default: 60 requests per minute
    "/api": int(
        os.getenv("RATE_LIMIT_API", "50")
    ),  # Default: 50 requests per minute for API endpoints
    "/settings": int(os.getenv("RATE_LIMIT_SETTINGS", "20")),  # Default: 20 requests per minute
    "/routing": int(os.getenv("RATE_LIMIT_ROUTING", "40")),  # Default: 40 requests per minute
}

DEFAULT_LIMIT = int(os.getenv("RATE_LIMIT_DEFAULT", "100"))  # Default: 100 requests per minute
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # Default: 60 seconds

# Method-specific limits (optional override)
METHOD_LIMITS = {
    "POST": int(os.getenv("RATE_LIMIT_POST", "0")),  # 0 means use endpoint-specific limits
    "PUT": int(os.getenv("RATE_LIMIT_PUT", "0")),
    "DELETE": int(os.getenv("RATE_LIMIT_DELETE", "0")),
}

logger = logging.getLogger("goblin_assistant")


def _middleware_order_debug_enabled() -> bool:
    return (os.getenv("MIDDLEWARE_ORDER_DEBUG", "0") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limits on API endpoints with proper headers.

    Kong Integration:
    - Checks for X-API-Gateway: Kong header to detect Kong-routed requests
    - Uses Kong's X-RateLimit-Remaining header when available
    - Applies secondary, more restrictive rate limiting for Kong-routed requests
    - Allows direct requests to bypass Kong's global limits
    """

    async def dispatch(self, request: Request, call_next):
        if _middleware_order_debug_enabled():
            correlation_id = getattr(request.state, "correlation_id", None)
            logger.info(
                "middleware_entry",
                extra={
                    "middleware": "RateLimitMiddleware",
                    "path": request.url.path,
                    "request_id": getattr(request.state, "request_id", None),
                    "correlation_id": correlation_id,
                    "correlation_source": "request_state" if correlation_id else "unset",
                },
            )

        # Skip rate limiting for metrics endpoint
        if request.url.path == "/metrics":
            return await call_next(request)

        # Check if request came through Kong Gateway
        is_kong_routed = request.headers.get("X-API-Gateway") == "Kong"
        kong_rate_remaining = request.headers.get("X-RateLimit-Remaining")

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # For Kong-routed requests, apply secondary rate limiting
        if is_kong_routed:
            # Kong has already done primary rate limiting
            # Apply more restrictive secondary limits for abuse prevention
            secondary_limit = DEFAULT_LIMIT // 10  # 10% of default limit
            endpoint_pattern = "kong_secondary"

            # Check Kong's remaining quota
            if kong_rate_remaining is not None:
                try:
                    kong_remaining = int(kong_rate_remaining)
                    # If Kong has low remaining quota, be more restrictive
                    if kong_remaining < 10:
                        secondary_limit = secondary_limit // 2
                except ValueError:
                    pass

            # Apply secondary rate limiting
            allowed, retry_after, remaining, reset_timestamp = limiter.is_allowed(
                client_ip, endpoint_pattern, secondary_limit, RATE_LIMIT_WINDOW
            )

            if not allowed:
                # Return rate limit exceeded with Kong coordination info
                response = Response(
                    content=f'{{"error": "rate_limit_exceeded", "message": "Secondary rate limit exceeded. Kong global limit may also be reached.", "retry_after": {retry_after}, "gateway": "kong", "secondary_limit": {secondary_limit}}}',
                    status_code=429,
                    media_type="application/json",
                )
                response.headers["Retry-After"] = str(retry_after)
                response.headers["X-RateLimit-Limit"] = str(secondary_limit)
                response.headers["X-RateLimit-Remaining"] = "0"
                response.headers["X-RateLimit-Reset"] = str(reset_timestamp)
                response.headers["X-API-Gateway"] = "Kong"
                return response

        # For direct requests (not through Kong), apply full rate limiting
        else:
            # Determine rate limit for this endpoint
            limit = DEFAULT_LIMIT
            endpoint_pattern = "direct"

            # Check method-specific limits first
            method_limit = METHOD_LIMITS.get(request.method)
            if method_limit and method_limit > 0:
                limit = method_limit
                endpoint_pattern = f"{request.method.lower()}_requests"

            # Then check endpoint-specific limits
            for pattern, pattern_limit in RATE_LIMITS.items():
                if request.url.path.startswith(pattern):
                    limit = pattern_limit
                    endpoint_pattern = pattern
                    break

            # Check rate limit
            allowed, retry_after, remaining, reset_timestamp = limiter.is_allowed(
                client_ip, endpoint_pattern, limit, RATE_LIMIT_WINDOW
            )

            if not allowed:
                # Return proper rate limit exceeded response with headers
                response = Response(
                    content=f'{{"error": "rate_limit_exceeded", "message": "Rate limit exceeded. Please try again later.", "retry_after": {retry_after}, "limit": {limit}, "window_seconds": {RATE_LIMIT_WINDOW}, "endpoint": "{endpoint_pattern}", "gateway": "direct"}}',
                    status_code=429,
                    media_type="application/json",
                )
                response.headers["Retry-After"] = str(retry_after)
                response.headers["X-RateLimit-Limit"] = str(limit)
                response.headers["X-RateLimit-Remaining"] = "0"
                response.headers["X-RateLimit-Reset"] = str(reset_timestamp)
                response.headers["X-RateLimit-Window-Seconds"] = str(RATE_LIMIT_WINDOW)
                response.headers["X-RateLimit-Endpoint"] = endpoint_pattern
                return response

        # Process request
        response = await call_next(request)

        # Add rate limit headers to successful responses
        if is_kong_routed:
            # For Kong requests, indicate secondary limiting was applied
            response.headers["X-API-Gateway"] = "Kong"
            response.headers["X-RateLimit-Secondary"] = "applied"
        else:
            # For direct requests, add standard rate limit headers
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(reset_timestamp)

        return response


def rate_limit_exceeded_handler(request: Request, exc: HTTPException):
    """Handler for rate limit exceptions (not needed with middleware approach)."""
    pass
