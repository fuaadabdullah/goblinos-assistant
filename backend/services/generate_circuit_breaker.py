"""Circuit-breaker and health-cache state for the generate pipeline.

Tracks per-provider failure counts, circuit-open timestamps, auth blocks,
and rate-limit cooldowns so the orchestration loop in ``generate_service``
can skip providers that are known-bad without burning latency budget.
"""

import os
import time

# ---------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------

_PROVIDER_HEALTH_TTL_S = float(os.getenv("GENERATE_PROVIDER_HEALTH_TTL_S", "60"))
_PROVIDER_CIRCUIT_FAILS = int(os.getenv("GENERATE_PROVIDER_CIRCUIT_FAILS", "3"))
_PROVIDER_CIRCUIT_COOLDOWN_S = float(
    os.getenv("GENERATE_PROVIDER_CIRCUIT_COOLDOWN_S", "30")
)
_PROVIDER_AUTH_COOLDOWN_S = float(os.getenv("GENERATE_PROVIDER_AUTH_COOLDOWN_S", "600"))
_PROVIDER_RATE_LIMIT_COOLDOWN_S = float(
    os.getenv("GENERATE_PROVIDER_RATE_LIMIT_COOLDOWN_S", "120")
)

# ---------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------

_provider_health_cache: dict[str, tuple[float, bool]] = {}
_provider_failure_counts: dict[str, int] = {}
_provider_circuit_open_until: dict[str, float] = {}
_provider_auth_blocked_until: dict[str, float] = {}
_provider_rate_limited_until: dict[str, float] = {}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def reset_provider_state() -> None:
    """Clear all provider circuit-breaker / health cache state."""
    _provider_health_cache.clear()
    _provider_failure_counts.clear()
    _provider_circuit_open_until.clear()
    _provider_auth_blocked_until.clear()
    _provider_rate_limited_until.clear()


def is_provider_blocked(provider_name: str) -> bool:
    return _provider_circuit_open_until.get(provider_name, 0.0) > time.time()


def is_provider_auth_blocked(provider_name: str) -> bool:
    return _provider_auth_blocked_until.get(provider_name, 0.0) > time.time()


def is_provider_rate_limited(provider_name: str) -> bool:
    return _provider_rate_limited_until.get(provider_name, 0.0) > time.time()


def provider_recently_unhealthy(provider_name: str) -> bool:
    cached = _provider_health_cache.get(provider_name)
    if not cached:
        return False
    ts, healthy = cached
    if (time.time() - ts) > _PROVIDER_HEALTH_TTL_S:
        return False
    return not healthy


def mark_provider_success(provider_name: str) -> None:
    _provider_health_cache[provider_name] = (time.time(), True)
    _provider_failure_counts[provider_name] = 0
    _provider_circuit_open_until.pop(provider_name, None)
    _provider_auth_blocked_until.pop(provider_name, None)
    _provider_rate_limited_until.pop(provider_name, None)


def mark_provider_failure(provider_name: str, retryable: bool = False) -> None:
    _provider_health_cache[provider_name] = (time.time(), False)
    if not retryable:
        return
    failures = _provider_failure_counts.get(provider_name, 0) + 1
    _provider_failure_counts[provider_name] = failures
    if failures >= _PROVIDER_CIRCUIT_FAILS:
        _provider_circuit_open_until[provider_name] = (
            time.time() + _PROVIDER_CIRCUIT_COOLDOWN_S
        )


def mark_provider_auth_failure(provider_name: str) -> None:
    _provider_health_cache[provider_name] = (time.time(), False)
    _provider_auth_blocked_until[provider_name] = (
        time.time() + _PROVIDER_AUTH_COOLDOWN_S
    )


def mark_provider_rate_limited(provider_name: str) -> None:
    _provider_health_cache[provider_name] = (time.time(), False)
    _provider_rate_limited_until[provider_name] = (
        time.time() + _PROVIDER_RATE_LIMIT_COOLDOWN_S
    )
