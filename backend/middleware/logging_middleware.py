"""Structured JSON logging middleware for production observability.

Features:
- JSON formatted logs for easy parsing
- Request/response logging with timing
- Error tracking with stack traces
- Correlation IDs for request tracing
- OpenTelemetry trace context integration
"""

import contextlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Try to import JSON logger, fall back to standard logging if not available
try:
    from pythonjsonlogger.json import JsonFormatter

    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False

# Try to import OpenTelemetry for trace context
try:
    from opentelemetry import trace

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False


def _middleware_order_debug_enabled() -> bool:
    return (os.getenv("MIDDLEWARE_ORDER_DEBUG", "0") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests/responses with structured JSON format."""

    def _setup_correlation_id(self, request: Request) -> tuple:
        """Setup correlation ID and return (id, id_at_entry, source, present)."""
        correlation_at_entry = getattr(request.state, "correlation_id", None)
        correlation_id_at_entry = str(correlation_at_entry or "").strip()
        correlation_present_at_entry = bool(correlation_id_at_entry)
        correlation_source = "request_state" if correlation_present_at_entry else "generated_fallback"
        correlation_id = correlation_id_at_entry if correlation_present_at_entry else str(uuid.uuid4())
        correlation_id_at_entry = correlation_id_at_entry if correlation_present_at_entry else "unset"
        request.state.correlation_id = correlation_id
        return correlation_id, correlation_id_at_entry, correlation_source, correlation_present_at_entry

    def _extract_trace_context(self) -> tuple:
        """Extract OpenTelemetry trace context if available."""
        if not HAS_OPENTELEMETRY:
            return None, None
        with contextlib.suppress(Exception):
            current_span = trace.get_current_span()
            if current_span and current_span.get_span_context().trace_id:
                trace_id = format(current_span.get_span_context().trace_id, "032x")
                span_id = format(current_span.get_span_context().span_id, "016x")
                return trace_id, span_id
        return None, None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Setup correlation ID
        correlation_id, correlation_id_at_entry, correlation_source, correlation_present_at_entry = (
            self._setup_correlation_id(request)
        )

        # Get request ID if available (from RequestIDMiddleware)
        request_id = getattr(request.state, "request_id", None)

        if _middleware_order_debug_enabled():
            logger.info(
                "middleware_entry",
                extra={
                    "middleware": "StructuredLoggingMiddleware",
                    "path": request.url.path,
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "correlation_id_at_entry": correlation_id_at_entry,
                    "correlation_source": correlation_source,
                    "correlation_present_at_entry": correlation_present_at_entry,
                },
            )

        # Start timer
        start_time = time.time()

        # Extract request body
        with contextlib.suppress(Exception):
            request.state.body = await request.body()

        # Extract trace context
        trace_id, span_id = self._extract_trace_context()

        # Build base log extra
        log_extra_base = {
            "request_id": request_id,
            "correlation_id": correlation_id,
            "correlation_id_at_entry": correlation_id_at_entry,
            "correlation_source": correlation_source,
            "correlation_present_at_entry": correlation_present_at_entry,
            "method": request.method,
            "path": request.url.path,
        }
        if trace_id:
            log_extra_base["trace_id"] = trace_id
            log_extra_base["span_id"] = span_id

        # Log incoming request
        log_extra = log_extra_base.copy()
        log_extra["query_params"] = dict(request.query_params)
        log_extra["client_host"] = request.client.host if request.client else None
        log_extra["user_agent"] = request.headers.get("user-agent")
        logger.info("incoming_request", extra=log_extra)

        # Process request
        try:
            response = await call_next(request)
            duration = time.time() - start_time

            # Log successful response
            log_extra = log_extra_base.copy()
            log_extra["status_code"] = response.status_code
            log_extra["duration_ms"] = round(duration * 1000, 2)
            logger.info("request_completed", extra=log_extra)

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            return response

        except Exception as e:
            duration = time.time() - start_time

            # Log error
            log_extra = log_extra_base.copy()
            log_extra["duration_ms"] = round(duration * 1000, 2)
            log_extra["error_type"] = type(e).__name__
            log_extra["error_message"] = str(e)
            logger.error("request_failed", extra=log_extra, exc_info=True)

            raise


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure structured JSON logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    # Create logger
    goblin_logger = logging.getLogger("goblin_assistant")
    goblin_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    goblin_logger.handlers.clear()

    # Create formatter - use JSON if available, otherwise standard
    if HAS_JSON_LOGGER:
        formatter = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
            },
        )
    else:
        # Fallback to standard formatter with JSON-like structure
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                # Add extra fields if present
                if hasattr(record, "request_id"):
                    log_entry["request_id"] = record.request_id
                if hasattr(record, "correlation_id"):
                    log_entry["correlation_id"] = record.correlation_id
                if hasattr(record, "method"):
                    log_entry["method"] = record.method
                if hasattr(record, "path"):
                    log_entry["path"] = record.path
                if hasattr(record, "status_code"):
                    log_entry["status_code"] = record.status_code
                if hasattr(record, "duration_ms"):
                    log_entry["duration_ms"] = record.duration_ms
                return json.dumps(log_entry)

        formatter = JSONFormatter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    goblin_logger.addHandler(console_handler)

    return goblin_logger


# Global logger instance
logger = setup_logging()
