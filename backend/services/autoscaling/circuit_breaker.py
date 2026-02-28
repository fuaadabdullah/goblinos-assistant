"""Circuit breaker helpers for autoscaling."""

import time

from .redis_store import redis_connection


async def check_circuit_breaker(service, provider_name: str) -> bool:
    """Check circuit breaker state for a provider."""
    async with redis_connection(service) as r:
        failure_key = f"circuit:{provider_name}:failures"
        state_key = f"circuit:{provider_name}:state"

        state = await r.get(state_key)
        if state == "open":
            opened_at = await r.get(f"circuit:{provider_name}:opened_at")
            if opened_at and time.time() - float(opened_at) > service.circuit_breaker_timeout:
                await r.delete(failure_key, state_key, f"circuit:{provider_name}:opened_at")
                return True
            return False

        return True


async def record_provider_failure(service, provider_name: str) -> None:
    """Record a provider failure for circuit breaker."""
    async with redis_connection(service) as r:
        failure_key = f"circuit:{provider_name}:failures"
        state_key = f"circuit:{provider_name}:state"

        failures = await r.incr(failure_key)
        await r.expire(failure_key, 300)

        if failures >= service.circuit_breaker_threshold:
            await r.setex(state_key, service.circuit_breaker_timeout, "open")
            await r.setex(
                f"circuit:{provider_name}:opened_at",
                service.circuit_breaker_timeout,
                str(time.time()),
            )


async def record_provider_success(service, provider_name: str) -> None:
    """Record a provider success (resets failure count)."""
    async with redis_connection(service) as r:
        failure_key = f"circuit:{provider_name}:failures"
        await r.delete(failure_key)
