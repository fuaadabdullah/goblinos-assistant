"""
Autoscaling Service for Goblin AI System
Handles rate limiting, spike detection, and graceful fallback routing.

Features:
- Rate limiting with sliding window
- Spike detection and automatic fallback
- Cheap model fallback (goblin-simple-llama-1b)
- Emergency auth/health fallback
- Circuit breaker pattern for provider failures
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from .autoscaling.circuit_breaker import (
    check_circuit_breaker,
    record_provider_failure,
    record_provider_success,
)
from .autoscaling.rate_limit import check_rate_limit
from .autoscaling.redis_store import close_redis, initialize_redis, redis_connection
from .autoscaling.types import AutoscalingMetrics, FallbackLevel, RateLimitConfig

logger = logging.getLogger(__name__)


class AutoscalingService:
    """Service for handling autoscaling, rate limiting, and fallback routing"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        config: RateLimitConfig | None = None,
    ):
        self.redis_url = redis_url
        self.config = config or RateLimitConfig()
        self.redis = None
        self._lock = asyncio.Lock()

        # Fallback model configuration
        self.cheap_fallback_model = "goblin-simple-llama-1b"
        self.emergency_endpoints = ["/health", "/auth/", "/api/status"]

        # Circuit breaker state
        self.circuit_breaker_failures: dict[str, int] = {}
        self.circuit_breaker_threshold = 5  # Failures before opening circuit
        self.circuit_breaker_timeout = 60  # Seconds to wait before retry

    async def initialize(self):
        """Initialize Redis connection"""
        await initialize_redis(self)
        logger.info("Autoscaling service initialized with Redis")

    async def close(self):
        """Close Redis connection"""
        await close_redis(self)

    @asynccontextmanager
    async def redis_connection(self):
        """Context manager for Redis operations"""
        try:
            async with redis_connection(self) as connection:
                yield connection
        except Exception as e:
            logger.error(f"Redis operation failed: {e}")
            raise

    async def check_rate_limit(
        self, client_ip: str, user_id: str | None = None
    ) -> tuple[bool, FallbackLevel, dict[str, Any]]:
        """
        Check if request should be rate limited and determine fallback level.

        Returns:
            (allowed, fallback_level, metadata)
        """
        allowed, fallback_level, metadata = await check_rate_limit(self, client_ip, user_id)

        identifier = user_id or client_ip
        request_count = metadata.get("request_count")
        spike_count = metadata.get("spike_count")
        cooldown_until = metadata.get("cooldown_until")

        if cooldown_until:
            logger.warning(f"Client {identifier} in cooldown until {cooldown_until}")
        elif spike_count and spike_count >= self.config.spike_threshold:
            logger.warning(
                f"Spike detected for {identifier}: {spike_count} requests in 10s, cooldown until {cooldown_until}"
            )
        elif not allowed and request_count is not None:
            logger.warning(f"Rate limit exceeded for {identifier}: {request_count} requests/minute")

        return allowed, fallback_level, metadata

    async def get_fallback_model(self, original_model: str, fallback_level: FallbackLevel) -> str:
        """Get the appropriate fallback model based on level"""
        if fallback_level == FallbackLevel.CHEAP_MODEL:
            return self.cheap_fallback_model
        elif fallback_level == FallbackLevel.EMERGENCY:
            return self.cheap_fallback_model
        else:
            return original_model

    async def is_emergency_endpoint(self, path: str) -> bool:
        """Check if endpoint should always be available (emergency fallback)"""
        return any(path.startswith(endpoint) for endpoint in self.emergency_endpoints)

    async def check_circuit_breaker(self, provider_name: str) -> bool:
        """
        Check circuit breaker state for a provider.
        Returns True if circuit is closed (requests allowed), False if open.
        """
        is_closed = await check_circuit_breaker(self, provider_name)
        if not is_closed:
            logger.warning(f"Circuit breaker open for {provider_name}")
        return is_closed

    async def record_provider_failure(self, provider_name: str):
        """Record a provider failure for circuit breaker"""
        await record_provider_failure(self, provider_name)

    async def record_provider_success(self, provider_name: str):
        """Record a provider success (resets failure count)"""
        await record_provider_success(self, provider_name)

    async def get_metrics(self) -> AutoscalingMetrics:
        """Get current autoscaling metrics"""
        async with self.redis_connection() as r:
            now = time.time()

            # Get global request rate (approximate)
            all_keys = await r.keys("requests:*")
            total_requests = 0
            for key in all_keys[:10]:  # Sample first 10 keys to avoid too many operations
                count = await r.zcount(key, now - 60, now)
                total_requests += count

            current_rpm = (total_requests / len(all_keys)) * 6 if all_keys else 0

            # Check for recent spikes
            spike_detected = False
            last_spike_time = None
            cooldown_keys = await r.keys("cooldown:*")
            if cooldown_keys:
                # Check if any cooldown is active
                for key in cooldown_keys[:5]:  # Sample first 5
                    cooldown_until = await r.get(key)
                    if cooldown_until and float(cooldown_until) > now:
                        spike_detected = True
                        # Get last spike time from zset
                        client_id = key.replace("cooldown:", "")
                        spike_times = await r.zrange(
                            f"requests:{client_id}", -1, -1, withscores=True
                        )
                        if spike_times:
                            last_spike_time = spike_times[0][1]

            # Determine current fallback level
            fallback_level = FallbackLevel.NORMAL
            if spike_detected:
                fallback_level = FallbackLevel.CHEAP_MODEL

            return AutoscalingMetrics(
                current_rpm=current_rpm,
                spike_detected=spike_detected,
                fallback_level=fallback_level,
                active_connections=len(all_keys),
                queue_depth=0,  # Would need queue system integration
                last_spike_time=last_spike_time,
            )

    async def graceful_shutdown(self):
        """Graceful shutdown - allow emergency endpoints only"""
        async with self.redis_connection() as r:
            await r.setex("system:shutdown", 3600, "true")  # 1 hour emergency mode
            logger.warning("System entering emergency mode - only auth/health endpoints available")

    async def is_emergency_mode(self) -> bool:
        """Check if system is in emergency mode"""
        async with self.redis_connection() as r:
            return bool(await r.exists("system:shutdown"))
