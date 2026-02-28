"""
License enforcement middleware for FastAPI.

Validates API keys and enforces licensing constraints on incoming requests.
"""

import os
import logging
from typing import Callable
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..licensing import get_license_validator, LicenseTier

logger = logging.getLogger("goblin_assistant")


def _middleware_order_debug_enabled() -> bool:
    return (os.getenv("MIDDLEWARE_ORDER_DEBUG", "0") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class LicenseEnforcementMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce license requirements on API requests.

    Checks for valid API key in:
    1. X-API-Key header
    2. Authorization header (Bearer token)
    3. Query parameter ?api_key=...

    Rejects requests without valid license (if enforcement enabled).
    Sets license info in request state for use in routers.
    """

    def __init__(self, app):
        super().__init__(app)
        self.validator = get_license_validator()
        self.enforce = self.validator.enforce_licensing

    async def dispatch(self, request: Request, call_next: Callable) -> Callable:
        """Process request with license validation"""
        if _middleware_order_debug_enabled():
            correlation_id = getattr(request.state, "correlation_id", None)
            logger.info(
                "middleware_entry",
                extra={
                    "middleware": "LicenseEnforcementMiddleware",
                    "path": request.url.path,
                    "request_id": getattr(request.state, "request_id", None),
                    "correlation_id": correlation_id,
                    "correlation_source": "request_state" if correlation_id else "unset",
                },
            )

        # Get API key from various sources
        api_key = self._extract_api_key(request)

        # Validate license
        is_valid, tier, error = self.validator.validate_license(api_key)

        if not is_valid and self.enforce:
            # Licensing is enforced and key is invalid
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": error or "Invalid or missing API key",
                    "error_code": "INVALID_LICENSE",
                },
            )

        # Store license info in request state for use in handlers
        tier_config = self.validator.get_tier_config(tier)
        request.state.license_tier = tier
        request.state.license_config = tier_config
        request.state.license_key = api_key or "default"

        # Add licensing headers to response
        response = await call_next(request)
        response.headers["X-License-Tier"] = tier.value
        response.headers["X-Rate-Limit-Per-Minute"] = str(tier_config["requests_per_minute"])

        return response

    def _extract_api_key(self, request: Request) -> str | None:
        """Extract API key from request"""

        # Check X-API-Key header
        if "x-api-key" in request.headers:
            return request.headers["x-api-key"]

        # Check Authorization header (Bearer token)
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]

        # Check query parameter
        if "api_key" in request.query_params:
            return request.query_params["api_key"]

        return None


class FeatureAccessMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce feature access based on license tier.

    Routes can be decorated with required features.
    """

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Callable:
        """Process request with feature checking"""
        if _middleware_order_debug_enabled():
            correlation_id = getattr(request.state, "correlation_id", None)
            logger.info(
                "middleware_entry",
                extra={
                    "middleware": "FeatureAccessMiddleware",
                    "path": request.url.path,
                    "request_id": getattr(request.state, "request_id", None),
                    "correlation_id": correlation_id,
                    "correlation_source": "request_state" if correlation_id else "unset",
                },
            )

        # Feature checking is done at route level via get_feature_access()
        return await call_next(request)


def get_license_info_from_request(request: Request):
    """Get license information from request state"""
    return {
        "tier": getattr(request.state, "license_tier", LicenseTier.FREE).value,
        "key": getattr(request.state, "license_key", "default"),
        "config": getattr(request.state, "license_config", {}),
    }


def require_feature(feature_name: str):
    """
    Decorator to require a specific feature for a route.

    Usage:
        @router.get("/rag-api")
        @require_feature("rag")
        async def rag_endpoint(request: Request):
            ...
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(request: Request, *args, **kwargs):
            # Get license tier from request state
            tier = getattr(request.state, "license_tier", LicenseTier.FREE)

            # Check if tier has access to feature
            validator = get_license_validator()
            if not validator.has_feature(tier, feature_name):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Feature '{feature_name}' requires a higher license tier",
                    headers={"X-Required-Tier": "pro"},
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
