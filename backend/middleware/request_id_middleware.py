"""Request ID middleware for generating unique request identifiers.

Features:
- Generates unique request IDs for each incoming request
- Adds request ID to request state and response headers
- Integrates with correlation ID for complete request tracing
"""

import logging
import os
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger("goblin_assistant")


def _middleware_order_debug_enabled() -> bool:
    return (os.getenv("MIDDLEWARE_ORDER_DEBUG", "0") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and attach unique request IDs to all requests."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Store IDs in request state before other middleware executes.
        request.state.request_id = request_id
        request.state.correlation_id = request_id

        if _middleware_order_debug_enabled():
            logger.info(
                "middleware_entry",
                extra={
                    "middleware": "RequestIDMiddleware",
                    "path": request.url.path,
                    "request_id": request_id,
                    "correlation_id": request_id,
                    "correlation_source": "request_id",
                },
            )

        # Process the request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response
