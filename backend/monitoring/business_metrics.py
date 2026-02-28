"""
Business metrics for Goblin Assistant.

Tracks key business KPIs including user engagement, costs, and provider usage.
"""

from typing import Optional
from opentelemetry import metrics
import time


class BusinessMetrics:
    """Tracks business-critical metrics for Goblin Assistant."""

    def __init__(self):
        self.meter = metrics.get_meter("goblin-assistant-business")

        # User engagement metrics
        self.active_users_daily = self.meter.create_counter(
            name="active_users_daily_total", description="Daily active users", unit="1"
        )

        self.active_users_monthly = self.meter.create_gauge(
            name="active_users_monthly_current",
            description="Current monthly active users",
            unit="1",
        )

        # Cost tracking metrics
        self.cost_per_request = self.meter.create_histogram(
            name="cost_per_request_dollars",
            description="Cost per API request in dollars",
            unit="USD",
        )

        self.total_cost_daily = self.meter.create_counter(
            name="total_cost_daily_dollars",
            description="Total daily costs across all providers",
            unit="USD",
        )

        # Provider usage metrics
        self.provider_requests_total = self.meter.create_counter(
            name="provider_requests_total",
            description="Total requests per provider",
            unit="1",
        )

        self.provider_tokens_total = self.meter.create_counter(
            name="provider_tokens_total",
            description="Total tokens used per provider",
            unit="1",
        )

        self.provider_latency = self.meter.create_histogram(
            name="provider_latency_seconds",
            description="Provider response latency",
            unit="s",
        )

        # Feature usage metrics
        self.feature_usage_total = self.meter.create_counter(
            name="feature_usage_total", description="Usage count per feature", unit="1"
        )

        # Error tracking by category
        self.business_errors_total = self.meter.create_counter(
            name="business_errors_total",
            description="Business logic errors by category",
            unit="1",
        )

    def record_user_activity(self, user_id: str, action: str = "request"):
        """Record user activity for DAU/MAU tracking."""
        # Use current day as a simple hash for daily aggregation
        day_key = int(time.time()) // 86400  # Days since epoch

        self.active_users_daily.add(
            1,
            {
                "action": action,
                "day": str(day_key),
                "user_id_hash": hash(user_id) % 1000000,  # Anonymized user ID
            },
        )

    def update_monthly_active_users(self, count: int):
        """Update the current MAU count."""
        self.active_users_monthly.set(count)

    def record_request_cost(
        self, cost_usd: float, provider: str, model: str, request_type: str = "chat"
    ):
        """Record the cost of an API request."""
        self.cost_per_request.record(
            cost_usd,
            {"provider": provider, "model": model, "request_type": request_type},
        )

        # Also accumulate daily total
        day_key = int(time.time()) // 86400
        self.total_cost_daily.add(cost_usd, {"provider": provider, "day": str(day_key)})

    def record_provider_usage(
        self,
        provider: str,
        model: str,
        tokens_used: Optional[int] = None,
        latency_seconds: Optional[float] = None,
        success: bool = True,
    ):
        """Record provider API usage."""
        self.provider_requests_total.add(
            1, {"provider": provider, "model": model, "success": str(success)}
        )

        if tokens_used is not None:
            self.provider_tokens_total.add(
                tokens_used,
                {
                    "provider": provider,
                    "model": model,
                    "token_type": "total",  # Could be split into prompt/completion
                },
            )

        if latency_seconds is not None:
            self.provider_latency.record(
                latency_seconds, {"provider": provider, "model": model}
            )

    def record_feature_usage(self, feature: str, user_id: Optional[str] = None):
        """Record usage of specific features."""
        attributes = {"feature": feature}
        if user_id:
            attributes["user_id_hash"] = hash(user_id) % 1000000

        self.feature_usage_total.add(1, attributes)

    def record_business_error(
        self, error_category: str, error_type: str, user_id: Optional[str] = None
    ):
        """Record business logic errors."""
        attributes = {"category": error_category, "error_type": error_type}
        if user_id:
            attributes["user_id_hash"] = hash(user_id) % 1000000

        self.business_errors_total.add(1, attributes)


# Global business metrics instance
business_metrics = BusinessMetrics()


# Convenience functions for easy usage throughout the codebase
def track_user_activity(user_id: str, action: str = "request"):
    """Track user activity for engagement metrics."""
    business_metrics.record_user_activity(user_id, action)


def track_request_cost(
    cost_usd: float, provider: str, model: str, request_type: str = "chat"
):
    """Track API request costs."""
    business_metrics.record_request_cost(cost_usd, provider, model, request_type)


def track_provider_usage(
    provider: str,
    model: str,
    tokens_used: Optional[int] = None,
    latency_seconds: Optional[float] = None,
    success: bool = True,
):
    """Track LLM provider usage."""
    business_metrics.record_provider_usage(
        provider, model, tokens_used, latency_seconds, success
    )


def track_feature_usage(feature: str, user_id: Optional[str] = None):
    """Track feature usage."""
    business_metrics.record_feature_usage(feature, user_id)


def track_business_error(
    error_category: str, error_type: str, user_id: Optional[str] = None
):
    """Track business logic errors."""
    business_metrics.record_business_error(error_category, error_type, user_id)
