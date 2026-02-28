"""
Standardized error handling using Problem Details RFC 7807.

This module provides consistent error responses across the API following
the Problem Details specification (RFC 7807).
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional

from fastapi import HTTPException
import httpx
from pydantic import BaseModel, Field

from backend.services.gateway_exceptions import TokenBudgetExceeded, MaxTokensExceeded
from backend.providers.circuit_breaker import CircuitBreakerOpen
from backend.providers.bulkhead import BulkheadExceeded

logger = logging.getLogger(__name__)


class ProblemDetail(BaseModel):
    """Problem Details response model following RFC 7807."""

    type: str = Field(
        default="about:blank",
        description="A URI reference that identifies the problem type",
    )
    title: str = Field(
        ..., description="A short, human-readable summary of the problem type"
    )
    detail: Optional[str] = Field(
        default=None,
        description="A human-readable explanation specific to this occurrence of the problem",
    )
    instance: Optional[str] = Field(
        default=None,
        description="A URI reference that identifies the specific occurrence of the problem",
    )
    status: int = Field(..., description="The HTTP status code")
    errors: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Field-specific validation errors"
    )
    code: Optional[str] = Field(
        default=None, description="Application-specific error code"
    )
    timestamp: Optional[str] = Field(
        default=None, description="ISO 8601 timestamp of when the error occurred"
    )


class ErrorCodes:
    """Standardized error codes for consistent error identification."""

    # Authentication & Authorization
    INVALID_API_KEY = "INVALID_API_KEY"
    MISSING_API_KEY = "MISSING_API_KEY"
    EXPIRED_TOKEN = "EXPIRED_TOKEN"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"

    # Validation
    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FIELD_VALUE = "INVALID_FIELD_VALUE"
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"

    # Business Logic
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # System
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    DATABASE_ERROR = "DATABASE_ERROR"


def _utc_now_iso_z() -> str:
    """Return an ISO 8601 UTC timestamp with trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_problem_detail(
    status: int,
    title: str,
    detail: Optional[str] = None,
    type_uri: Optional[str] = None,
    code: Optional[str] = None,
    errors: Optional[Dict[str, List[str]]] = None,
    instance: Optional[str] = None,
) -> ProblemDetail:
    """
    Create a standardized ProblemDetail response.

    Args:
        status: HTTP status code
        title: Human-readable summary of the problem
        detail: Specific explanation of this occurrence
        type_uri: URI identifying the problem type
        code: Application-specific error code
        errors: Field-specific validation errors
        instance: URI identifying this specific occurrence

    Returns:
        ProblemDetail object
    """
    problem = ProblemDetail(
        type=type_uri or "about:blank",
        title=title,
        detail=detail,
        status=status,
        code=code,
        errors=errors,
        instance=instance,
        timestamp=_utc_now_iso_z(),
    )

    return problem


def raise_problem(
    status: int,
    title: str,
    detail: Optional[str] = None,
    type_uri: Optional[str] = None,
    code: Optional[str] = None,
    errors: Optional[Dict[str, List[str]]] = None,
    instance: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """
    Raise an HTTPException with ProblemDetail content.

    This function creates a standardized problem detail response and raises
    an HTTPException that FastAPI will serialize as JSON.
    """
    problem = create_problem_detail(
        status=status,
        title=title,
        detail=detail,
        type_uri=type_uri,
        code=code,
        errors=errors,
        instance=instance,
    )

    raise HTTPException(
        status_code=status, detail=problem.model_dump(), headers=headers
    )


# Convenience functions for common error types


def raise_validation_error(
    detail: str,
    errors: Optional[Dict[str, List[str]]] = None,
    instance: Optional[str] = None,
) -> None:
    """Raise a validation error (400)."""
    raise_problem(
        status=400,
        title="Validation Error",
        detail=detail,
        type_uri="https://goblin-backend.onrender.com/errors/validation",
        code=ErrorCodes.INVALID_REQUEST,
        errors=errors,
        instance=instance,
    )


def raise_unauthorized(
    detail: str = "Authentication required", instance: Optional[str] = None
) -> None:
    """Raise an unauthorized error (401)."""
    raise_problem(
        status=401,
        title="Unauthorized",
        detail=detail,
        type_uri="https://goblin-backend.onrender.com/errors/unauthorized",
        code=ErrorCodes.INVALID_API_KEY,
        instance=instance,
    )


def raise_forbidden(
    detail: str = "Insufficient permissions", instance: Optional[str] = None
) -> None:
    """Raise a forbidden error (403)."""
    raise_problem(
        status=403,
        title="Forbidden",
        detail=detail,
        type_uri="https://goblin-backend.onrender.com/errors/forbidden",
        code=ErrorCodes.INSUFFICIENT_PERMISSIONS,
        instance=instance,
    )


def raise_not_found(resource: str, instance: Optional[str] = None) -> None:
    """Raise a not found error (404)."""
    raise_problem(
        status=404,
        title="Not Found",
        detail=f"{resource} not found",
        type_uri="https://goblin-backend.onrender.com/errors/not-found",
        instance=instance,
    )


def raise_rate_limited(
    detail: str = "Rate limit exceeded", instance: Optional[str] = None
) -> None:
    """Raise a rate limit exceeded error (429)."""
    raise_problem(
        status=429,
        title="Too Many Requests",
        detail=detail,
        type_uri="https://goblin-backend.onrender.com/errors/rate-limited",
        code=ErrorCodes.RATE_LIMIT_EXCEEDED,
        instance=instance,
    )


def raise_internal_error(
    detail: str = "An internal error occurred", instance: Optional[str] = None
) -> None:
    """Raise an internal server error (500)."""
    logger.error("Internal error", extra={"detail": detail, "instance": instance})
    raise_problem(
        status=500,
        title="Internal Server Error",
        detail=detail,
        type_uri="https://goblin-backend.onrender.com/errors/internal",
        code=ErrorCodes.INTERNAL_ERROR,
        instance=instance,
    )


def raise_service_unavailable(service: str, instance: Optional[str] = None) -> None:
    """Raise a service unavailable error (503)."""
    raise_problem(
        status=503,
        title="Service Unavailable",
        detail=f"{service} is currently unavailable",
        type_uri="https://goblin-backend.onrender.com/errors/service-unavailable",
        code=ErrorCodes.SERVICE_UNAVAILABLE,
        instance=instance,
    )


def _map_http_exception(
    exc: HTTPException, correlation_id: Optional[str]
) -> ProblemDetail:
    """Map HTTPException to ProblemDetail."""
    if isinstance(exc.detail, dict):
        # If the caller already raised a ProblemDetail-shaped dict, preserve it.
        if "title" in exc.detail and "status" in exc.detail:
            try:
                return ProblemDetail.model_validate(exc.detail)
            except Exception:
                # Fall back to creating a ProblemDetail below.
                pass
    return create_problem_detail(
        status=exc.status_code,
        title="Request failed",
        detail=str(exc.detail),
        code=None,
        instance=correlation_id,
    )


def _map_token_budget_exceeded(
    exc: TokenBudgetExceeded, correlation_id: Optional[str]
) -> ProblemDetail:
    """Map TokenBudgetExceeded to ProblemDetail."""
    return create_problem_detail(
        status=400,
        title="Token budget exceeded",
        detail=str(exc),
        type_uri="https://goblin-backend.onrender.com/errors/token-budget",
        code=ErrorCodes.QUOTA_EXCEEDED,
        instance=correlation_id,
    )


def _map_max_tokens_exceeded(
    exc: MaxTokensExceeded, correlation_id: Optional[str]
) -> ProblemDetail:
    """Map MaxTokensExceeded to ProblemDetail."""
    return create_problem_detail(
        status=400,
        title="Max tokens exceeded",
        detail=str(exc),
        type_uri="https://goblin-backend.onrender.com/errors/max-tokens",
        code=ErrorCodes.INVALID_REQUEST,
        instance=correlation_id,
    )


def _map_circuit_breaker_open(
    exc: CircuitBreakerOpen, correlation_id: Optional[str]
) -> ProblemDetail:
    """Map CircuitBreakerOpen to ProblemDetail."""
    return create_problem_detail(
        status=503,
        title="Provider temporarily unavailable",
        detail=str(exc),
        type_uri="https://goblin-backend.onrender.com/errors/circuit-open",
        code=ErrorCodes.SERVICE_UNAVAILABLE,
        instance=correlation_id,
    )


def _map_bulkhead_exceeded(
    exc: BulkheadExceeded, correlation_id: Optional[str]
) -> ProblemDetail:
    """Map BulkheadExceeded to ProblemDetail."""
    return create_problem_detail(
        status=503,
        title="Provider overloaded",
        detail=str(exc),
        type_uri="https://goblin-backend.onrender.com/errors/provider-overloaded",
        code=ErrorCodes.SERVICE_UNAVAILABLE,
        instance=correlation_id,
    )


def _map_timeout_exception(
    exc: httpx.TimeoutException, correlation_id: Optional[str]
) -> ProblemDetail:
    """Map timeout exception to ProblemDetail."""
    # Note: exc parameter is not used but kept for consistency with other mappers
    del exc  # Make it explicit that we're not using this parameter
    return create_problem_detail(
        status=503,
        title="Provider timeout",
        detail="Upstream provider timed out",
        type_uri="https://goblin-backend.onrender.com/errors/provider-timeout",
        code=ErrorCodes.SERVICE_UNAVAILABLE,
        instance=correlation_id,
    )


def _map_generic_exception(
    exc: Exception, correlation_id: Optional[str]
) -> ProblemDetail:
    """Map generic exception to ProblemDetail."""
    return create_problem_detail(
        status=500,
        title="Internal Server Error",
        detail=str(exc),
        type_uri="https://goblin-backend.onrender.com/errors/internal",
        code=ErrorCodes.INTERNAL_ERROR,
        instance=correlation_id,
    )


def map_exception_to_problem(
    exc: Exception, correlation_id: Optional[str] = None
) -> ProblemDetail:
    """Map known exceptions to standardized ProblemDetails."""
    if isinstance(exc, HTTPException):
        return _map_http_exception(exc, correlation_id)
    if isinstance(exc, TokenBudgetExceeded):
        return _map_token_budget_exceeded(exc, correlation_id)
    if isinstance(exc, MaxTokensExceeded):
        return _map_max_tokens_exceeded(exc, correlation_id)
    if isinstance(exc, CircuitBreakerOpen):
        return _map_circuit_breaker_open(exc, correlation_id)
    if isinstance(exc, BulkheadExceeded):
        return _map_bulkhead_exceeded(exc, correlation_id)
    if isinstance(exc, httpx.TimeoutException):
        return _map_timeout_exception(exc, correlation_id)
    return _map_generic_exception(exc, correlation_id)
