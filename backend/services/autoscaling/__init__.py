"""Autoscaling helper modules."""

from .types import AutoscalingMetrics, FallbackLevel, RateLimitConfig
from .rate_limit import check_rate_limit
from .circuit_breaker import (
    check_circuit_breaker,
    record_provider_failure,
    record_provider_success,
)
from .redis_store import (
    close_redis,
    get_redis_connection,
    initialize_redis,
    redis_connection,
)

__all__ = [
    "AutoscalingMetrics",
    "FallbackLevel",
    "RateLimitConfig",
    "check_rate_limit",
    "check_circuit_breaker",
    "record_provider_failure",
    "record_provider_success",
    "close_redis",
    "get_redis_connection",
    "initialize_redis",
    "redis_connection",
]
