"""
Unified metrics collection for Goblin Assistant.

Combines Prometheus metrics with OpenTelemetry for comprehensive observability.
"""

from typing import Optional, Dict
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from opentelemetry import metrics as otel_metrics

# OpenTelemetry meter
otel_meter = otel_metrics.get_meter("goblin-assistant-prometheus")

# HTTP Metrics
http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests in progress",
    ["method", "endpoint"],
)

# Business Metrics
chat_completions_total = Counter(
    "chat_completions_total", "Total chat completions", ["provider", "model"]
)

chat_completion_tokens_total = Counter(
    "chat_completion_tokens_total",
    "Total tokens used in chat completions",
    ["provider", "model", "token_type"],
)

provider_latency_seconds = Histogram(
    "provider_latency_seconds",
    "Provider response latency",
    ["provider", "operation"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# Error Metrics
http_errors_total = Counter(
    "http_errors_total", "Total HTTP errors", ["method", "endpoint", "error_type"]
)

business_errors_total = Counter(
    "business_errors_total", "Total business logic errors", ["category", "error_type"]
)

# Health Metrics
service_health_status = Gauge(
    "service_health_status",
    "Service health status (1=healthy, 0=unhealthy)",
    ["service_name"],
)

# OpenTelemetry equivalents for unified export
otel_http_requests_total = otel_meter.create_counter(
    name="http_requests_total", description="Total HTTP requests", unit="1"
)

otel_http_request_duration = otel_meter.create_histogram(
    name="http_request_duration_seconds",
    description="HTTP request duration in seconds",
    unit="s",
)

otel_chat_completions_total = otel_meter.create_counter(
    name="chat_completions_total", description="Total chat completions", unit="1"
)

otel_business_errors_total = otel_meter.create_counter(
    name="business_errors_total", description="Total business logic errors", unit="1"
)


class MetricsCollector:
    """Unified metrics collector that updates both Prometheus and OpenTelemetry."""

    @staticmethod
    def record_http_request(
        method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record an HTTP request."""
        # Prometheus metrics
        http_requests_total.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()

        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
            duration
        )

        # OpenTelemetry metrics
        otel_http_requests_total.add(
            1, {"method": method, "endpoint": endpoint, "status_code": str(status_code)}
        )

        otel_http_request_duration.record(
            duration, {"method": method, "endpoint": endpoint}
        )

    @staticmethod
    def record_chat_completion(
        provider: str, model: str, tokens_used: Optional[Dict[str, int]] = None
    ):
        """Record a chat completion."""
        # Prometheus metrics
        chat_completions_total.labels(provider=provider, model=model).inc()

        if tokens_used:
            for token_type, count in tokens_used.items():
                chat_completion_tokens_total.labels(
                    provider=provider, model=model, token_type=token_type
                ).inc(count)

        # OpenTelemetry metrics
        attributes = {"provider": provider, "model": model}
        otel_chat_completions_total.add(1, attributes)

    @staticmethod
    def record_provider_latency(provider: str, operation: str, latency: float):
        """Record provider latency."""
        # Prometheus metrics
        provider_latency_seconds.labels(provider=provider, operation=operation).observe(
            latency
        )

    @staticmethod
    def record_http_error(method: str, endpoint: str, error_type: str):
        """Record an HTTP error."""
        http_errors_total.labels(
            method=method, endpoint=endpoint, error_type=error_type
        ).inc()

    @staticmethod
    def record_business_error(category: str, error_type: str):
        """Record a business logic error."""
        # Prometheus metrics
        business_errors_total.labels(category=category, error_type=error_type).inc()

        # OpenTelemetry metrics
        otel_business_errors_total.add(
            1, {"category": category, "error_type": error_type}
        )

    @staticmethod
    def set_service_health(service_name: str, healthy: bool):
        """Set service health status."""
        service_health_status.labels(service_name=service_name).set(1 if healthy else 0)

    @staticmethod
    def get_prometheus_metrics():
        """Get Prometheus metrics in the format expected by /metrics endpoint."""
        return generate_latest()


# Global metrics collector instance
metrics_collector = MetricsCollector()


# Convenience functions for easy usage
def record_http_request(method: str, endpoint: str, status_code: int, duration: float):
    """Record an HTTP request with both Prometheus and OpenTelemetry."""
    metrics_collector.record_http_request(method, endpoint, status_code, duration)


def record_chat_completion(
    provider: str, model: str, tokens_used: Optional[Dict[str, int]] = None
):
    """Record a chat completion."""
    metrics_collector.record_chat_completion(provider, model, tokens_used)


def record_provider_latency(provider: str, operation: str, latency: float):
    """Record provider latency."""
    metrics_collector.record_provider_latency(provider, operation, latency)


def record_business_error(category: str, error_type: str):
    """Record a business logic error."""
    metrics_collector.record_business_error(category, error_type)


def set_service_health(service_name: str, healthy: bool):
    """Set service health status."""
    metrics_collector.set_service_health(service_name, healthy)
