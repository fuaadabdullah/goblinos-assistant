"""
Sentry monitoring configuration for Goblin Assistant backend.

Provides error tracking, performance monitoring, and crash reporting.
Integrated with OpenTelemetry for unified observability.
"""

import os

# Try to import sentry_sdk - make it optional
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration

    HAS_SENTRY = True
except ImportError:
    HAS_SENTRY = False
    sentry_sdk = None
    FastApiIntegration = None

try:
    from sentry_sdk.integrations.sqlalchemy import SqlAlchemyIntegration
except Exception:
    SqlAlchemyIntegration = None

try:
    from sentry_sdk.integrations.redis import RedisIntegration
except Exception:
    RedisIntegration = None

# Try to import OpenTelemetry for integration
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False


def init_sentry():
    """Initialize Sentry monitoring for the backend."""
    if not HAS_SENTRY:
        print("ℹ️  Sentry not initialized (sentry_sdk not installed)")
        return

    sentry_dsn = (os.getenv("SENTRY_DSN") or "").strip()
    environment = os.getenv("ENVIRONMENT", "development")
    has_valid_dsn_scheme = sentry_dsn.startswith("http://") or sentry_dsn.startswith("https://")

    if sentry_dsn and not has_valid_dsn_scheme:
        print("⚠️  Sentry DSN is set but invalid; skipping Sentry initialization")
        return

    if sentry_dsn and has_valid_dsn_scheme and environment in ["staging", "production"]:
        sentry_sdk.init(
            dsn=sentry_dsn,
            environment=environment,
            integrations=[
                FastApiIntegration(
                    transaction_style="endpoint",
                    http_methods_to_capture=["GET", "POST", "PUT", "DELETE", "PATCH"],
                ),
                *([SqlAlchemyIntegration()] if SqlAlchemyIntegration else []),
                *([RedisIntegration()] if RedisIntegration else []),
            ],
            # Performance monitoring
            traces_sample_rate=0.1,  # Capture 10% of transactions for performance
            profiles_sample_rate=0.1,  # Capture 10% of profiles
            # Error tracking
            send_default_pii=False,  # Don't send personally identifiable information
            attach_stacktrace=True,
            # Release tracking
            release=os.getenv("RELEASE_VERSION"),
            dist=os.getenv("RELEASE_DIST"),
            # Before send hook for filtering
            before_send=before_send,
        )
        print(f"✅ Sentry initialized for {environment} environment")
    else:
        print("ℹ️  Sentry not initialized (missing DSN or not production)")


def before_send(event, hint):
    """Filter events before sending to Sentry."""
    # Filter out common non-errors
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]
        if exc_type:
            # Filter out client disconnects (common in web apps)
            if "ClientDisconnected" in str(exc_type):
                return None
            # Filter out validation errors from user input
            if (
                "ValidationError" in str(exc_type)
                and "pydantic" in str(exc_value).lower()
            ):
                return None

    # Add OpenTelemetry trace context if available
    if HAS_OPENTELEMETRY:
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.get_span_context().trace_id:
                trace_id = format(current_span.get_span_context().trace_id, "032x")
                span_id = format(current_span.get_span_context().span_id, "016x")

                # Add trace context to Sentry event
                if "tags" not in event:
                    event["tags"] = {}
                event["tags"]["trace_id"] = trace_id
                event["tags"]["span_id"] = span_id

                # Set span status to ERROR if we have an exception
                if "exc_info" in hint:
                    current_span.set_status(
                        Status(StatusCode.ERROR, str(hint["exc_info"][1]))
                    )

        except Exception:
            # Don't let OpenTelemetry integration break Sentry
            pass

    return event


def capture_exception(error: Exception, **kwargs):
    """Capture an exception with additional context."""
    sentry_sdk.capture_exception(error, **kwargs)


def capture_message(message: str, level: str = "info", **kwargs):
    """Capture a message with additional context."""
    sentry_sdk.capture_message(message, level=level, **kwargs)


def set_user_context(user_id: str, email: str = None, **kwargs):
    """Set user context for error tracking."""
    sentry_sdk.set_user({"id": user_id, "email": email, **kwargs})


def set_tag(key: str, value: str):
    """Set a tag for error tracking."""
    sentry_sdk.set_tag(key, value)


def add_breadcrumb(
    message: str, category: str = "default", level: str = "info", **kwargs
):
    """Add a breadcrumb for debugging."""
    sentry_sdk.add_breadcrumb(message=message, category=category, level=level, **kwargs)
