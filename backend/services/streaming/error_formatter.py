"""Formatter for streaming errors."""

from typing import Any


class StreamErrorFormatter:
    """Formatter for streaming errors."""

    def format_validation_error(self, errors: list[str]) -> dict[str, Any]:
        """Format validation errors."""
        return {
            "type": "validation_error",
            "message": "Request validation failed",
            "errors": errors,
        }

    def format_rate_limit_error(self, retry_after: float | None) -> dict[str, Any]:
        """Format rate limit errors."""
        error_msg = {"type": "rate_limit_exceeded", "message": "Rate limit exceeded"}
        if retry_after:
            error_msg["retry_after"] = retry_after
        return error_msg

    def format_provider_error(self, error_msg: str) -> dict[str, Any]:
        """Format provider errors."""
        return {
            "type": "provider_error",
            "message": error_msg,
        }
