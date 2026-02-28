"""Rate limiting helpers for autoscaling."""

import time
from typing import Any

from .redis_store import redis_connection
from .types import FallbackLevel


async def check_rate_limit(
    service,
    client_ip: str,
    user_id: str | None = None,
) -> tuple[bool, FallbackLevel, dict[str, Any]]:
    """Check if request should be rate limited and determine fallback level."""
    async with service._lock:
        async with redis_connection(service) as r:
            identifier = user_id or client_ip
            now = time.time()

            await r.zremrangebyscore(f"requests:{identifier}", 0, now - 60)

            request_count = await r.zcard(f"requests:{identifier}")
            spike_count = await r.zcount(f"requests:{identifier}", now - 10, now)

            fallback_level = FallbackLevel.NORMAL
            allowed = True
            metadata = {
                "request_count": request_count,
                "spike_count": spike_count,
                "cooldown_until": None,
            }

            cooldown_key = f"cooldown:{identifier}"
            cooldown_until = await r.get(cooldown_key)

            if cooldown_until and float(cooldown_until) > now:
                fallback_level = FallbackLevel.CHEAP_MODEL
                metadata["cooldown_until"] = float(cooldown_until)
            elif spike_count >= service.config.spike_threshold:
                cooldown_until = now + (service.config.cooldown_minutes * 60)
                await r.setex(
                    cooldown_key,
                    service.config.cooldown_minutes * 60,
                    str(cooldown_until),
                )
                fallback_level = FallbackLevel.CHEAP_MODEL
                metadata["cooldown_until"] = cooldown_until
            elif request_count >= service.config.requests_per_minute:
                allowed = False
                fallback_level = FallbackLevel.DENY_REQUEST

            if allowed:
                await r.zadd(f"requests:{identifier}", {str(now): now})
                await r.expire(f"requests:{identifier}", 120)

            return allowed, fallback_level, metadata
