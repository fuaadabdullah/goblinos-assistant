"""
Security Headers Middleware for FastAPI
Provides security headers similar to Helmet.js for production hardening
"""

import logging
import os
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger("goblin_assistant")


def _middleware_order_debug_enabled() -> bool:
    return (os.getenv("MIDDLEWARE_ORDER_DEBUG", "0") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        if _middleware_order_debug_enabled():
            correlation_id = getattr(request.state, "correlation_id", None)
            logger.info(
                "middleware_entry",
                extra={
                    "middleware": "SecurityHeadersMiddleware",
                    "path": request.url.path,
                    "request_id": getattr(request.state, "request_id", None),
                    "correlation_id": correlation_id,
                    "correlation_source": "request_state" if correlation_id else "unset",
                },
            )

        response = await call_next(request)

        # Content Security Policy - restrict resource loading
        # Note: 'unsafe-inline' is required for style-src due to Next.js inline styles.
        # script-src avoids 'unsafe-eval'; if a library requires it, use a hash or nonce instead.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https: wss:; "
            "frame-ancestors 'none';"
        )

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Enable XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions Policy (formerly Feature Policy)
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )

        # HSTS - only in production with HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        return response
