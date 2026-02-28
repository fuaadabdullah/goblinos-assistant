"""Types for autoscaling helpers."""

from dataclasses import dataclass
from enum import Enum


class FallbackLevel(Enum):
    """Fallback levels for graceful degradation."""

    NORMAL = "normal"
    CHEAP_MODEL = "cheap_model"
    DENY_REQUEST = "deny"
    EMERGENCY = "emergency"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 100
    burst_limit: int = 20
    spike_threshold: int = 50
    cooldown_minutes: int = 5


@dataclass
class AutoscalingMetrics:
    """Current autoscaling metrics."""

    current_rpm: float
    spike_detected: bool
    fallback_level: FallbackLevel
    active_connections: int
    queue_depth: int
    last_spike_time: float | None
