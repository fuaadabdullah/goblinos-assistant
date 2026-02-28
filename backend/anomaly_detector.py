"""
Anomaly detection and alerting for token usage and system metrics.

Monitors for unusual patterns in:
- Token usage spikes
- High error rates
- Unusual request patterns
- Budget exhaustion
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    TOKEN_SPIKE = "token_spike"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    HIGH_ERROR_RATE = "high_error_rate"
    UNUSUAL_PATTERN = "unusual_pattern"
    SYSTEM_OVERLOAD = "system_overload"


@dataclass
class Alert:
    """Alert data structure."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: float
    api_key: Optional[str] = None
    resolved: bool = False


class MetricsCollector:
    """Collects and analyzes metrics for anomaly detection."""

    def __init__(self, redis_client=None, window_size_minutes: int = 60):
        self.redis = redis_client
        self.window_size = window_size_minutes * 60  # Convert to seconds

        # Rolling windows for metrics
        self.token_usage_window: List[Dict[str, Any]] = []
        self.error_rate_window: List[Dict[str, Any]] = []
        self.request_pattern_window: List[Dict[str, Any]] = []

    async def record_token_usage(
        self, api_key: str, tokens: int, timestamp: Optional[float] = None
    ):
        """Record token usage for analysis."""
        if timestamp is None:
            timestamp = time.time()

        usage_data = {"api_key": api_key, "tokens": tokens, "timestamp": timestamp}

        self.token_usage_window.append(usage_data)
        self._cleanup_old_data(self.token_usage_window)

        # Store in Redis if available
        if self.redis:
            try:
                key = f"metrics:token_usage:{api_key}"
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.redis.lpush(key, str(usage_data))
                )
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.redis.expire(key, self.window_size)
                )
            except Exception as e:
                logger.warning(f"Failed to store token usage metrics: {e}")

    async def record_request_error(
        self, api_key: Optional[str], error_type: str, timestamp: Optional[float] = None
    ):
        """Record request errors for analysis."""
        if timestamp is None:
            timestamp = time.time()

        error_data = {
            "api_key": api_key,
            "error_type": error_type,
            "timestamp": timestamp,
        }

        self.error_rate_window.append(error_data)
        self._cleanup_old_data(self.error_rate_window)

    async def record_request_pattern(
        self, api_key: str, intent: str, tokens: int, timestamp: Optional[float] = None
    ):
        """Record request patterns for analysis."""
        if timestamp is None:
            timestamp = time.time()

        pattern_data = {
            "api_key": api_key,
            "intent": intent,
            "tokens": tokens,
            "timestamp": timestamp,
        }

        self.request_pattern_window.append(pattern_data)
        self._cleanup_old_data(self.request_pattern_window)

    def _cleanup_old_data(self, data_list: List[Dict[str, Any]]):
        """Remove data older than the window size."""
        cutoff_time = time.time() - self.window_size
        while data_list and data_list[0]["timestamp"] < cutoff_time:
            data_list.pop(0)

    def detect_token_spike(self, api_key: str, current_tokens: int) -> Optional[Alert]:
        """Detect unusual token usage spikes."""
        # Get recent usage for this API key
        recent_usage = [
            item for item in self.token_usage_window if item["api_key"] == api_key
        ]

        if len(recent_usage) < 5:  # Need minimum data points
            return None

        # Calculate statistics
        token_values = [item["tokens"] for item in recent_usage[:-1]]  # Exclude current
        if not token_values:
            return None

        try:
            mean_usage = statistics.mean(token_values)
            stdev_usage = statistics.stdev(token_values) if len(token_values) > 1 else 0

            # Check if current usage is 3+ standard deviations above mean
            if stdev_usage > 0 and current_tokens > mean_usage + (3 * stdev_usage):
                return Alert(
                    alert_id=f"token_spike_{api_key}_{int(time.time())}",
                    alert_type=AlertType.TOKEN_SPIKE,
                    severity=AlertSeverity.HIGH,
                    message=f"Unusual token usage spike detected for API key {api_key}",
                    details={
                        "api_key": api_key,
                        "current_tokens": current_tokens,
                        "mean_usage": mean_usage,
                        "stdev_usage": stdev_usage,
                        "threshold": mean_usage + (3 * stdev_usage),
                    },
                    timestamp=time.time(),
                    api_key=api_key,
                )
        except statistics.StatisticsError:
            pass

        return None

    def detect_budget_exhaustion(
        self, api_key: str, usage_stats: Dict[str, Any]
    ) -> Optional[Alert]:
        """Detect when API key is approaching budget exhaustion."""
        usage_percentage = usage_stats.get("usage_percentage", 0)

        if usage_percentage > 90:
            severity = (
                AlertSeverity.CRITICAL if usage_percentage > 95 else AlertSeverity.HIGH
            )
            return Alert(
                alert_id=f"budget_exhaustion_{api_key}_{int(time.time())}",
                alert_type=AlertType.BUDGET_EXHAUSTION,
                severity=severity,
                message=f"API key {api_key} approaching budget exhaustion",
                details={
                    "api_key": api_key,
                    "usage_percentage": usage_percentage,
                    "current_usage": usage_stats.get("current_usage"),
                    "budget_limit": usage_stats.get("budget_limit"),
                },
                timestamp=time.time(),
                api_key=api_key,
            )

        return None

    def detect_high_error_rate(
        self, api_key: Optional[str] = None, time_window_minutes: int = 10
    ) -> Optional[Alert]:
        """Detect high error rates."""
        cutoff_time = time.time() - (time_window_minutes * 60)

        # Filter errors by API key and time window
        relevant_errors = [
            error
            for error in self.error_rate_window
            if error["timestamp"] > cutoff_time
            and (api_key is None or error["api_key"] == api_key)
        ]

        # Count total requests in the same window (approximate)
        total_requests = len(
            [
                req
                for req in self.request_pattern_window
                if req["timestamp"] > cutoff_time
                and (api_key is None or req["api_key"] == api_key)
            ]
        )

        if total_requests < 10:  # Need minimum sample size
            return None

        error_rate = len(relevant_errors) / total_requests

        if error_rate > 0.5:  # 50% error rate threshold
            severity = (
                AlertSeverity.CRITICAL if error_rate > 0.8 else AlertSeverity.HIGH
            )
            return Alert(
                alert_id=f"high_error_rate_{api_key or 'global'}_{int(time.time())}",
                alert_type=AlertType.HIGH_ERROR_RATE,
                severity=severity,
                message=f"High error rate detected: {error_rate:.1%}",
                details={
                    "api_key": api_key,
                    "error_rate": error_rate,
                    "error_count": len(relevant_errors),
                    "total_requests": total_requests,
                    "time_window_minutes": time_window_minutes,
                },
                timestamp=time.time(),
                api_key=api_key,
            )

        return None

    def detect_unusual_patterns(self, api_key: str) -> Optional[Alert]:
        """Detect unusual request patterns."""
        recent_requests = [
            req for req in self.request_pattern_window if req["api_key"] == api_key
        ]

        if len(recent_requests) < 10:
            return None

        # Check for unusual intent distribution
        intents = [req["intent"] for req in recent_requests[-20:]]  # Last 20 requests
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # Flag if 80%+ of requests are the same intent (potential abuse)
        total_intents = len(intents)
        for intent, count in intent_counts.items():
            if count / total_intents > 0.8:
                return Alert(
                    alert_id=f"unusual_pattern_{api_key}_{int(time.time())}",
                    alert_type=AlertType.UNUSUAL_PATTERN,
                    severity=AlertSeverity.MEDIUM,
                    message=f"Unusual request pattern detected for API key {api_key}",
                    details={
                        "api_key": api_key,
                        "dominant_intent": intent,
                        "percentage": count / total_intents,
                        "intent_distribution": intent_counts,
                    },
                    timestamp=time.time(),
                    api_key=api_key,
                )

        return None


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers = []

    def add_alert_handler(self, handler):
        """Add a handler function for alerts."""
        self.alert_handlers.append(handler)

    async def process_alert(self, alert: Alert):
        """Process and distribute an alert."""
        # Store alert
        self.active_alerts[alert.alert_id] = alert

        # Store in Redis if available
        if self.redis:
            try:
                key = f"alerts:active:{alert.alert_id}"
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.redis.set(
                        key, str(alert.__dict__), ex=86400
                    ),  # 24 hours
                )
            except Exception as e:
                logger.warning(f"Failed to store alert: {e}")

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.warning(f"ALERT {alert.severity.value.upper()}: {alert.message}")

    async def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True

            # Remove from Redis
            if self.redis:
                try:
                    key = f"alerts:active:{alert_id}"
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.redis.delete(key)
                    )
                except Exception as e:
                    logger.warning(f"Failed to resolve alert: {e}")

    def get_active_alerts(self, api_key: Optional[str] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by API key."""
        alerts = list(self.active_alerts.values())
        if api_key:
            alerts = [a for a in alerts if a.api_key == api_key]
        return [a for a in alerts if not a.resolved]


class AnomalyDetector:
    """Main anomaly detection service."""

    def __init__(self, redis_client=None):
        self.metrics = MetricsCollector(redis_client)
        self.alert_manager = AlertManager(redis_client)

        # Default alert handler (logs alerts)
        self.alert_manager.add_alert_handler(self._default_alert_handler)

    async def _default_alert_handler(self, alert: Alert):
        """Default alert handler that logs alerts."""
        logger.warning(
            f"ANOMALY DETECTED: {alert.alert_type.value} - {alert.message}",
            extra={
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "api_key": alert.api_key,
                "details": alert.details,
            },
        )

    async def analyze_request(
        self,
        api_key: Optional[str],
        tokens_used: int,
        intent: str,
        success: bool,
        error_type: Optional[str] = None,
    ):
        """Analyze a completed request for anomalies."""

        # Record metrics
        if api_key:
            await self.metrics.record_token_usage(api_key, tokens_used)
            await self.metrics.record_request_pattern(api_key, intent, tokens_used)

        if not success and error_type:
            await self.metrics.record_request_error(api_key, error_type)

        # Check for anomalies
        alerts = []

        if api_key:
            # Token spike detection
            spike_alert = self.metrics.detect_token_spike(api_key, tokens_used)
            if spike_alert:
                alerts.append(spike_alert)

            # Unusual pattern detection
            pattern_alert = self.metrics.detect_unusual_patterns(api_key)
            if pattern_alert:
                alerts.append(pattern_alert)

        # High error rate detection
        error_alert = self.metrics.detect_high_error_rate(api_key)
        if error_alert:
            alerts.append(error_alert)

        # Process alerts
        for alert in alerts:
            await self.alert_manager.process_alert(alert)

    async def check_budget_alerts(self, api_key: str, usage_stats: Dict[str, Any]):
        """Check for budget-related alerts."""
        budget_alert = self.metrics.detect_budget_exhaustion(api_key, usage_stats)
        if budget_alert:
            await self.alert_manager.process_alert(budget_alert)

    def get_active_alerts(self, api_key: Optional[str] = None) -> List[Alert]:
        """Get active alerts."""
        return self.alert_manager.get_active_alerts(api_key)


# Global anomaly detector instance
_anomaly_detector = None


def get_anomaly_detector(redis_client=None) -> AnomalyDetector:
    """Get or create anomaly detector instance."""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector(redis_client)
    return _anomaly_detector
