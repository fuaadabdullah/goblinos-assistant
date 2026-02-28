"""
OpenTelemetry configuration for unified observability.

Provides centralized tracing, metrics, and logging instrumentation across:
- FastAPI application
- HTTP clients (httpx)
- Database operations (SQLAlchemy)
- Redis operations
- External LLM provider calls

Exports to OTLP collector for integration with observability backends.
"""

import os
import logging

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.distro import distro

    HAS_OPENTELEMETRY = True
except Exception:
    # OpenTelemetry not available in test environment - degrade gracefully
    trace = None
    metrics = None
    TracerProvider = None
    BatchSpanProcessor = None
    MeterProvider = None
    PeriodicExportingMetricReader = None
    Resource = None
    OTLPSpanExporter = None
    OTLPMetricExporter = None
    FastAPIInstrumentor = None
    HTTPXClientInstrumentor = None
    SQLAlchemyInstrumentor = None
    RedisInstrumentor = None
    distro = None
    HAS_OPENTELEMETRY = False


def create_resource():
    """Create OpenTelemetry resource with service information."""
    return Resource.create(
        {
            "service.name": "goblin-assistant-backend",
            "service.version": os.getenv("RELEASE_VERSION", "1.0.0"),
            "service.environment": os.getenv("ENVIRONMENT", "development"),
            "service.instance.id": os.getenv("INSTANCE_ID", "local"),
        }
    )


def init_opentelemetry():
    """Initialize OpenTelemetry SDK with OTLP exporters."""

    environment = os.getenv("ENVIRONMENT", "development")
    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4318")

    # Only initialize in production/staging environments or when explicitly disabled
    if os.getenv("DISABLE_OPENTELEMETRY", "").lower() == "true":
        print("ℹ️  OpenTelemetry disabled (DISABLE_OPENTELEMETRY=true)")
        return

    # Check if OpenTelemetry is available
    if not HAS_OPENTELEMETRY:
        print("ℹ️  OpenTelemetry not available - skipping initialization")
        return

    try:
        # Initialize distro (sets up global providers)
        distro.configure()

        # Create resource
        resource = create_resource()

        # Configure tracing
        trace.set_tracer_provider(TracerProvider(resource=resource))

        # Add OTLP trace exporter
        otlp_trace_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=otlp_endpoint.startswith(
                "http://"
            ),  # Use insecure for local development
        )
        span_processor = BatchSpanProcessor(otlp_trace_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        # Configure metrics
        metric_exporter = OTLPMetricExporter(
            endpoint=otlp_endpoint,
            insecure=otlp_endpoint.startswith("http://"),
        )
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=15000,  # Export every 15 seconds
        )
        metrics.set_meter_provider(
            MeterProvider(
                resource=resource,
                metric_readers=[metric_reader],
            )
        )

        # Auto-instrument libraries
        _setup_auto_instrumentation()

        print(f"✅ OpenTelemetry initialized for {environment} environment")
        print(f"   OTLP Endpoint: {otlp_endpoint}")

    except Exception as e:
        print(f"❌ Failed to initialize OpenTelemetry: {e}")
        logging.exception("OpenTelemetry initialization failed")


def _setup_auto_instrumentation():
    """Configure auto-instrumentation for key libraries."""

    # FastAPI instrumentation (will be applied when app starts)
    # Note: We don't call FastAPIInstrumentor().instrument() here
    # as it needs to be done after the FastAPI app is created

    # HTTPX client instrumentation
    try:
        HTTPXClientInstrumentor().instrument()
        print("✅ HTTPX instrumentation enabled")
    except Exception as e:
        print(f"⚠️  HTTPX instrumentation failed: {e}")

    # SQLAlchemy instrumentation
    try:
        SQLAlchemyInstrumentor().instrument(
            engine=None,  # Instrument all engines
            service="goblin-assistant-db",
        )
        print("✅ SQLAlchemy instrumentation enabled")
    except Exception as e:
        print(f"⚠️  SQLAlchemy instrumentation failed: {e}")

    # Redis instrumentation
    try:
        RedisInstrumentor().instrument()
        print("✅ Redis instrumentation enabled")
    except Exception as e:
        print(f"⚠️  Redis instrumentation failed: {e}")


def instrument_fastapi_app(app):
    """Instrument a FastAPI application with OpenTelemetry."""
    if not HAS_OPENTELEMETRY or FastAPIInstrumentor is None:
        print("ℹ️  OpenTelemetry not available - skipping FastAPI instrumentation")
        return
    try:
        FastAPIInstrumentor().instrument_app(app)
        print("✅ FastAPI instrumentation enabled")
    except Exception as e:
        print(f"⚠️  FastAPI instrumentation failed: {e}")


def get_tracer(name: str = "goblin-assistant"):
    """Get a tracer instance for manual instrumentation."""
    return trace.get_tracer(name)


def get_meter(name: str = "goblin-assistant"):
    """Get a meter instance for custom metrics."""
    return metrics.get_meter(name)
