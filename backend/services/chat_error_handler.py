"""
ChatErrorHandler Service for handling chat errors.

This service handles all error management for chat completion requests,
separating error handling concerns from the main chat handler.
"""

import logging
import time
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)


class ErrorHandlerConfig:
    """Backward-compatible config placeholder for ChatErrorHandler tests."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ChatErrorHandler:
    """Service for handling chat errors."""

    def __init__(self):
        """Initialize the ChatErrorHandler."""
        self.error_counts: Dict[str, int] = {}
        self.last_error_time: Dict[str, float] = {}

    def handle_chat_error(self, error: Exception, request_id: str) -> Dict[str, Any]:
        """
        Handle chat errors and return error response.

        Args:
            error: The exception that occurred
            request_id: Request ID for tracking

        Returns:
            Formatted error response
        """
        try:
            # Determine error type and message
            error_type = type(error).__name__
            error_message = str(error)

            # Log the error
            self._log_error(error, request_id)

            # Categorize the error
            error_category = self._categorize_error(error)

            # Build error response
            error_response = self._build_error_response(
                error_type, error_message, error_category, request_id
            )

            # Track error metrics
            self._track_error_metrics(error_type, request_id)

            return error_response

        except Exception as e:
            # Fallback error handling
            logger.error(f"Error in error handler: {e}")
            return self._build_fallback_error_response(request_id)

    async def handle_validation_error(
        self, error: Exception, request_id: str, session_id: str
    ) -> Dict[str, Any]:
        """
        Handle validation errors (test-compatible signature).

        Args:
            error: The validation exception
            request_id: Request ID for tracking
            session_id: Session ID for tracking

        Returns:
            Formatted validation error response
        """
        import asyncio

        await asyncio.sleep(0)  # Make truly async for tests
        try:
            error_response = {
                "error": {
                    "type": "validation_error",
                    "message": str(error),
                },
                "error_type": "validation_error",
                "message": str(error),
                "request_id": request_id,
                "session_id": session_id,
                "timestamp": int(time.time()),
            }

            logger.warning(
                f"Validation error for request {request_id}, session {session_id}: {error}"
            )
            return error_response

        except Exception as e:
            logger.error(f"Error handling validation error: {e}")
            return self._build_fallback_error_response(request_id)

    def handle_rate_limit_error(
        self, retry_after: Optional[float], request_id: str
    ) -> Dict[str, Any]:
        """
        Handle rate limit errors.

        Args:
            retry_after: Time to wait before retrying (in seconds)
            request_id: Request ID for tracking

        Returns:
            Formatted rate limit error response
        """
        try:
            error_response = {
                "id": request_id,
                "object": "error",
                "error": {
                    "type": "rate_limit_exceeded",
                    "message": "Rate limit exceeded",
                    "code": 429,
                    "timestamp": int(time.time()),
                },
            }

            if retry_after is not None:
                error_response["error"]["retry_after"] = retry_after

            logger.warning(
                f"Rate limit exceeded for request {request_id}, retry after: {retry_after}"
            )
            return error_response

        except Exception as e:
            logger.error(f"Error handling rate limit error: {e}")
            return self._build_fallback_error_response(request_id)

    async def handle_provider_error(
        self, error: Exception, request_id: str, provider_name: str
    ) -> Dict[str, Any]:
        """
        Handle provider errors (test-compatible signature).

        Args:
            error: The provider exception
            request_id: Request ID for tracking
            provider_name: Name of the provider

        Returns:
            Formatted provider error response
        """
        import asyncio

        await asyncio.sleep(0)  # Make truly async for tests
        try:
            error_response = {
                "error": {
                    "type": "provider_error",
                    "message": str(error),
                },
                "error_type": "provider_error",
                "message": str(error),
                "request_id": request_id,
                "provider_name": provider_name,
                "timestamp": int(time.time()),
            }

            logger.error(
                f"Provider error for request {request_id}, provider {provider_name}: {error}"
            )
            return error_response

        except Exception as e:
            logger.error(f"Error handling provider error: {e}")
            return self._build_fallback_error_response(request_id)

    def handle_timeout_error(
        self, timeout_duration: float, request_id: str
    ) -> Dict[str, Any]:
        """
        Handle timeout errors.

        Args:
            timeout_duration: Duration that caused timeout (in seconds)
            request_id: Request ID for tracking

        Returns:
            Formatted timeout error response
        """
        try:
            error_response = {
                "id": request_id,
                "object": "error",
                "error": {
                    "type": "timeout_error",
                    "message": f"Request timed out after {timeout_duration} seconds",
                    "code": 408,
                    "timeout_duration": timeout_duration,
                    "timestamp": int(time.time()),
                },
            }

            logger.warning(
                f"Timeout error for request {request_id} after {timeout_duration}s"
            )
            return error_response

        except Exception as e:
            logger.error(f"Error handling timeout error: {e}")
            return self._build_fallback_error_response(request_id)

    def handle_network_error(
        self, network_error: str, request_id: str
    ) -> Dict[str, Any]:
        """
        Handle network errors.

        Args:
            network_error: Network error message
            request_id: Request ID for tracking

        Returns:
            Formatted network error response
        """
        try:
            error_response = {
                "id": request_id,
                "object": "error",
                "error": {
                    "type": "network_error",
                    "message": f"Network error: {network_error}",
                    "code": 502,
                    "timestamp": int(time.time()),
                },
            }

            logger.error(f"Network error for request {request_id}: {network_error}")
            return error_response

        except Exception as e:
            logger.error(f"Error handling network error: {e}")
            return self._build_fallback_error_response(request_id)

    def handle_authentication_error(
        self, auth_error: str, request_id: str
    ) -> Dict[str, Any]:
        """
        Handle authentication errors.

        Args:
            auth_error: Authentication error message
            request_id: Request ID for tracking

        Returns:
            Formatted authentication error response
        """
        try:
            error_response = {
                "id": request_id,
                "object": "error",
                "error": {
                    "type": "authentication_error",
                    "message": f"Authentication failed: {auth_error}",
                    "code": 401,
                    "timestamp": int(time.time()),
                },
            }

            logger.error(f"Authentication error for request {request_id}: {auth_error}")
            return error_response

        except Exception as e:
            logger.error(f"Error handling authentication error: {e}")
            return self._build_fallback_error_response(request_id)

    def handle_authorization_error(
        self, authz_error: str, request_id: str
    ) -> Dict[str, Any]:
        """
        Handle authorization errors.

        Args:
            authz_error: Authorization error message
            request_id: Request ID for tracking

        Returns:
            Formatted authorization error response
        """
        try:
            error_response = {
                "id": request_id,
                "object": "error",
                "error": {
                    "type": "authorization_error",
                    "message": f"Authorization failed: {authz_error}",
                    "code": 403,
                    "timestamp": int(time.time()),
                },
            }

            logger.error(f"Authorization error for request {request_id}: {authz_error}")
            return error_response

        except Exception as e:
            logger.error(f"Error handling authorization error: {e}")
            return self._build_fallback_error_response(request_id)

    def handle_internal_error(
        self, internal_error: str, request_id: str
    ) -> Dict[str, Any]:
        """
        Handle internal server errors.

        Args:
            internal_error: Internal error message
            request_id: Request ID for tracking

        Returns:
            Formatted internal error response
        """
        try:
            error_response = {
                "id": request_id,
                "object": "error",
                "error": {
                    "type": "internal_error",
                    "message": f"Internal server error: {internal_error}",
                    "code": 500,
                    "timestamp": int(time.time()),
                },
            }

            logger.error(f"Internal error for request {request_id}: {internal_error}")
            return error_response

        except Exception as e:
            logger.error(f"Error handling internal error: {e}")
            return self._build_fallback_error_response(request_id)

    def _log_error(self, error: Exception, request_id: str) -> None:
        """Log the error with appropriate level."""
        error_type = type(error).__name__

        if error_type in ["ValidationError", "ValueError", "TypeError"]:
            logger.warning(f"Validation error in request {request_id}: {error}")
        elif error_type in ["RateLimitError", "TooManyRequestsError"]:
            logger.warning(f"Rate limit error in request {request_id}: {error}")
        elif error_type in ["TimeoutError", "asyncio.TimeoutError"]:
            logger.warning(f"Timeout error in request {request_id}: {error}")
        elif error_type in ["ConnectionError", "NetworkError"]:
            logger.error(f"Network error in request {request_id}: {error}")
        elif error_type in ["AuthenticationError", "UnauthorizedError"]:
            logger.error(f"Authentication error in request {request_id}: {error}")
        elif error_type in ["AuthorizationError", "ForbiddenError"]:
            logger.error(f"Authorization error in request {request_id}: {error}")
        elif error_type in ["ProviderError", "ServiceUnavailableError"]:
            logger.error(f"Provider error in request {request_id}: {error}")
        else:
            logger.error(f"Unexpected error in request {request_id}: {error}")

    def _categorize_error(self, error: Exception) -> str:
        """Categorize the error type."""
        error_type = type(error).__name__

        if error_type in ["ValidationError", "ValueError", "TypeError"]:
            return "validation"
        elif error_type in ["RateLimitError", "TooManyRequestsError"]:
            return "rate_limit"
        elif error_type in ["TimeoutError", "asyncio.TimeoutError"]:
            return "timeout"
        elif error_type in ["ConnectionError", "NetworkError"]:
            return "network"
        elif error_type in ["AuthenticationError", "UnauthorizedError"]:
            return "authentication"
        elif error_type in ["AuthorizationError", "ForbiddenError"]:
            return "authorization"
        elif error_type in ["ProviderError", "ServiceUnavailableError"]:
            return "provider"
        else:
            return "internal"

    def _build_error_response(
        self, error_type: str, error_message: str, error_category: str, request_id: str
    ) -> Dict[str, Any]:
        """Build a standardized error response."""
        # Map error categories to HTTP status codes
        status_code_map = {
            "validation": 400,
            "rate_limit": 429,
            "timeout": 408,
            "network": 502,
            "authentication": 401,
            "authorization": 403,
            "provider": 503,
            "internal": 500,
        }

        status_code = status_code_map.get(error_category, 500)

        error_response = {
            "id": request_id,
            "object": "error",
            "error": {
                "type": error_type,
                "message": error_message,
                "code": status_code,
                "category": error_category,
                "timestamp": int(time.time()),
            },
        }

        # Add additional context for debugging (in development)
        import os

        if os.getenv("DEBUG", "false").lower() == "true":
            import traceback

            error_response["error"]["traceback"] = traceback.format_exc()

        return error_response

    def _build_fallback_error_response(self, request_id: str) -> Dict[str, Any]:
        """Build a fallback error response when error handling itself fails."""
        return {
            "id": request_id,
            "object": "error",
            "error": {
                "type": "fallback_error",
                "message": "An unexpected error occurred while handling the original error",
                "code": 500,
                "timestamp": int(time.time()),
            },
        }

    def _track_error_metrics(self, error_type: str, request_id: str) -> None:
        """Track error metrics for monitoring."""
        current_time = time.time()

        # Update error count
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        # Update last error time
        self.last_error_time[error_type] = current_time

        # Log metrics periodically (could be sent to monitoring system)
        if self.error_counts[error_type] % 10 == 0:  # Log every 10 errors
            logger.info(
                f"Error metrics - {error_type}: {self.error_counts[error_type]} errors"
            )

    def get_error_metrics(self) -> Dict[str, Any]:
        """Get current error metrics."""
        return {
            "error_counts": self.error_counts.copy(),
            "last_error_times": self.last_error_time.copy(),
            "total_errors": sum(self.error_counts.values()),
        }

    def reset_error_metrics(self) -> None:
        """Reset error metrics (useful for testing)."""
        self.error_counts.clear()
        self.last_error_time.clear()

    def is_error_recoverable(self, error: Exception) -> bool:
        """
        Determine if an error is recoverable (can be retried).

        Args:
            error: The exception to check

        Returns:
            True if the error is recoverable, False otherwise
        """
        error_type = type(error).__name__

        # Recoverable errors (can be retried)
        recoverable_errors = [
            "TimeoutError",
            "asyncio.TimeoutError",
            "ConnectionError",
            "NetworkError",
            "ServiceUnavailableError",
            "ProviderError",
        ]

        # Non-recoverable errors (should not be retried)
        non_recoverable_errors = [
            "ValidationError",
            "ValueError",
            "TypeError",
            "AuthenticationError",
            "UnauthorizedError",
            "AuthorizationError",
            "ForbiddenError",
        ]

        if error_type in recoverable_errors:
            return True
        elif error_type in non_recoverable_errors:
            return False
        else:
            # For unknown errors, be conservative and don't retry
            return False

    def should_rate_limit_user(self, user_id: Optional[str], error_type: str) -> bool:
        """
        Determine if a user should be rate limited based on error patterns.

        Args:
            user_id: User ID to check
            error_type: Type of error that occurred

        Returns:
            True if user should be rate limited, False otherwise
        """
        if not user_id:
            return False

        # Rate limit users who repeatedly cause validation errors
        if error_type in ["ValidationError", "ValueError", "TypeError"]:
            # This would typically check a database or cache
            # For now, return False as placeholder
            return False

        return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get error handling statistics.

        Returns:
            Dict containing error statistics
        """
        return {
            "total_errors": sum(self.error_counts.values()),
            "validation_errors": self.error_counts.get("validation_error", 0),
            "provider_errors": self.error_counts.get("provider_error", 0),
            "system_errors": self.error_counts.get("system_error", 0),
            "timestamp": time.time(),
        }
