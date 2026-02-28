"""
ChatErrorFormatter Service for formatting chat errors.

This service handles error formatting for chat completion requests,
separating error formatting concerns from the main chat handler.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Types of error severity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Types of error categories."""

    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    PROVIDER = "provider"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorFormatConfig:
    """Configuration for error formatting."""

    include_stack_trace: bool = False
    include_request_context: bool = True
    include_timestamp: bool = True
    include_error_code: bool = True
    include_severity: bool = True
    include_request_id: bool = True
    include_category: bool = True
    sanitize_sensitive_data: bool = True
    max_error_message_length: int = 500
    enable_error_categorization: bool = True


@dataclass
class FormattedError:
    """Formatted error with metadata."""

    error_message: str
    error_type: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: float
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    suggested_actions: Optional[List[str]] = None


class ChatErrorFormatter:
    """Service for formatting chat errors."""

    def __init__(self, config: Optional[ErrorFormatConfig] = None):
        """Initialize the ChatErrorFormatter."""
        self.config = config or ErrorFormatConfig()

    def format_error(
        self,
        error: Exception,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> FormattedError:
        """
        Format an error.

        Args:
            error: Exception to format
            request_id: Request ID for tracking
            session_id: Session ID for tracking
            context: Additional context information

        Returns:
            Formatted error
        """
        try:
            # Determine error category and severity
            category, severity = self._categorize_error(error)

            # Generate error code
            error_code = self._generate_error_code(error, category)

            # Format error message
            error_message = self._format_error_message(error)

            # Extract stack trace if enabled
            stack_trace = None
            if self.config.include_stack_trace:
                import traceback

                stack_trace = traceback.format_exc()

            # Build context
            error_context = self._build_error_context(context, request_id, session_id)

            # Get suggested actions
            suggested_actions = self._get_suggested_actions(error, category, severity)

            # Create formatted error
            formatted_error = FormattedError(
                error_message=error_message,
                error_type=type(error).__name__,
                severity=severity,
                category=category,
                timestamp=time.time(),
                request_id=request_id,
                session_id=session_id,
                error_code=error_code,
                stack_trace=stack_trace,
                context=error_context,
                suggested_actions=suggested_actions,
            )

            logger.debug(f"Formatted error for request {request_id}: {error_code}")
            return formatted_error

        except Exception as e:
            logger.error(f"Error formatting error: {e}")
            # Return fallback error format
            return self._create_fallback_error(error, request_id, session_id)

    def format_validation_error(
        self,
        validation_errors: List[Dict[str, Any]],
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> FormattedError:
        """
        Format a validation error.

        Args:
            validation_errors: List of validation errors
            request_id: Request ID for tracking
            session_id: Session ID for tracking

        Returns:
            Formatted validation error
        """
        try:
            # Build error message from validation errors
            error_messages = [err.get("message", str(err)) for err in validation_errors]
            error_message = f"Validation failed: {'; '.join(error_messages)}"

            # Create context with validation details
            context = {
                "validation_errors": validation_errors,
                "error_count": len(validation_errors),
            }

            return self.format_error(
                ValueError(error_message),
                request_id,
                session_id,
                context,
            )

        except Exception as e:
            logger.error(f"Error formatting validation error: {e}")
            return self._create_fallback_error(
                ValueError(str(e)), request_id, session_id
            )

    def format_provider_error(
        self,
        provider_name: str,
        provider_error: Exception,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> FormattedError:
        """
        Format a provider error.

        Args:
            provider_name: Name of the provider
            provider_error: Provider-specific error
            request_id: Request ID for tracking
            session_id: Session ID for tracking

        Returns:
            Formatted provider error
        """
        try:
            # Create context with provider information
            context = {
                "provider_name": provider_name,
                "provider_error_type": type(provider_error).__name__,
                "provider_error_message": str(provider_error),
            }

            formatted_error = self.format_error(
                provider_error,
                request_id,
                session_id,
                context,
            )

            # Override category to PROVIDER for provider errors
            formatted_error.category = ErrorCategory.PROVIDER

            return formatted_error

        except Exception as e:
            logger.error(f"Error formatting provider error: {e}")
            return self._create_fallback_error(
                ValueError(str(e)), request_id, session_id
            )

    def format_timeout_error(
        self,
        timeout_type: str,
        timeout_seconds: float,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> FormattedError:
        """
        Format a timeout error.

        Args:
            timeout_type: Type of timeout
            timeout_seconds: Timeout duration in seconds
            request_id: Request ID for tracking
            session_id: Session ID for tracking

        Returns:
            Formatted timeout error
        """
        try:
            error_message = (
                f"Request timed out after {timeout_seconds} seconds ({timeout_type})"
            )

            # Create context with timeout information
            context = {
                "timeout_type": timeout_type,
                "timeout_seconds": timeout_seconds,
            }

            return self.format_error(
                TimeoutError(error_message),
                request_id,
                session_id,
                context,
            )

        except Exception as e:
            logger.error(f"Error formatting timeout error: {e}")
            return self._create_fallback_error(
                TimeoutError(str(e)), request_id, session_id
            )

    def format_rate_limit_error(
        self,
        limit_type: str,
        limit_value: Union[int, float],
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> FormattedError:
        """
        Format a rate limit error.

        Args:
            limit_type: Type of rate limit
            limit_value: Rate limit value
            request_id: Request ID for tracking
            session_id: Session ID for tracking

        Returns:
            Formatted rate limit error
        """
        try:
            error_message = f"Rate limit exceeded: {limit_type} ({limit_value})"

            # Create context with rate limit information
            context = {
                "limit_type": limit_type,
                "limit_value": limit_value,
            }

            return self.format_error(
                ValueError(error_message),
                request_id,
                session_id,
                context,
            )

        except Exception as e:
            logger.error(f"Error formatting rate limit error: {e}")
            return self._create_fallback_error(
                ValueError(str(e)), request_id, session_id
            )

    def get_error_response_headers(self) -> Dict[str, str]:
        """Get HTTP headers for error responses."""
        return {
            "Content-Type": "application/json",
            "X-Error-Format": "structured",
        }

    def update_error_config(self, config: ErrorFormatConfig) -> None:
        """Update error formatting configuration."""
        self.config = config
        logger.info("Updated error formatting configuration")

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error formatting statistics."""
        try:
            return {
                "config": {
                    "include_stack_trace": self.config.include_stack_trace,
                    "include_request_context": self.config.include_request_context,
                    "include_timestamp": self.config.include_timestamp,
                    "include_error_code": self.config.include_error_code,
                    "include_severity": self.config.include_severity,
                    "include_category": self.config.include_category,
                    "sanitize_sensitive_data": self.config.sanitize_sensitive_data,
                    "max_error_message_length": self.config.max_error_message_length,
                    "enable_error_categorization": self.config.enable_error_categorization,
                },
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error getting error stats: {e}")
            return {"error": str(e)}

    def reset_stats(self) -> None:
        """Reset error formatting statistics."""
        try:
            logger.info("Reset error formatting statistics")

        except Exception as e:
            logger.error(f"Error resetting error stats: {e}")

    def _categorize_error(
        self, error: Exception
    ) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Categorize an error and determine its severity."""
        try:
            error_type = type(error).__name__.lower()

            # Categorization rules
            if "validation" in error_type or "value" in error_type:
                category = ErrorCategory.VALIDATION
                severity = ErrorSeverity.MEDIUM
            elif "auth" in error_type:
                if "not authenticated" in str(error).lower():
                    category = ErrorCategory.AUTHENTICATION
                    severity = ErrorSeverity.HIGH
                else:
                    category = ErrorCategory.AUTHORIZATION
                    severity = ErrorSeverity.HIGH
            elif "rate" in error_type or "limit" in error_type:
                category = ErrorCategory.RATE_LIMIT
                severity = ErrorSeverity.MEDIUM
            elif "timeout" in error_type:
                category = ErrorCategory.TIMEOUT
                severity = ErrorSeverity.HIGH
            elif "connection" in error_type or "network" in error_type:
                category = ErrorCategory.NETWORK
                severity = ErrorSeverity.HIGH
            elif "provider" in error_type or "service" in error_type:
                category = ErrorCategory.PROVIDER
                severity = ErrorSeverity.HIGH
            elif "system" in error_type or "internal" in error_type:
                category = ErrorCategory.SYSTEM
                severity = ErrorSeverity.CRITICAL
            else:
                category = ErrorCategory.UNKNOWN
                severity = ErrorSeverity.MEDIUM

            return category, severity

        except Exception as e:
            logger.error(f"Error categorizing error: {e}")
            return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM

    def _generate_error_code(self, error: Exception, category: ErrorCategory) -> str:
        """Generate a unique error code."""
        try:
            # Create error code based on category and error type
            category_prefix = {
                ErrorCategory.VALIDATION: "VAL",
                ErrorCategory.AUTHENTICATION: "AUTH",
                ErrorCategory.AUTHORIZATION: "AUTHZ",
                ErrorCategory.RATE_LIMIT: "RATE",
                ErrorCategory.TIMEOUT: "TIME",
                ErrorCategory.PROVIDER: "PROV",
                ErrorCategory.NETWORK: "NET",
                ErrorCategory.SYSTEM: "SYS",
                ErrorCategory.UNKNOWN: "UNK",
            }

            prefix = category_prefix.get(category, "ERR")
            error_type = type(error).__name__[:3].upper()
            timestamp = str(int(time.time()))[-6:]  # Last 6 digits of timestamp

            return f"{prefix}-{error_type}-{timestamp}"

        except Exception as e:
            logger.error(f"Error generating error code: {e}")
            return f"ERR-UNK-{int(time.time())}"

    def _format_error_message(self, error: Exception) -> str:
        """Format the error message."""
        try:
            message = str(error)

            # Truncate if too long
            if len(message) > self.config.max_error_message_length:
                message = message[: self.config.max_error_message_length] + "..."

            # Sanitize sensitive data if enabled
            if self.config.sanitize_sensitive_data:
                message = self._sanitize_message(message)

            return message

        except Exception as e:
            logger.error(f"Error formatting error message: {e}")
            return "An error occurred"

    def _sanitize_message(self, message: str) -> str:
        """Sanitize error message to remove sensitive data."""
        try:
            # Remove common sensitive patterns
            import re

            # Remove API keys/tokens
            message = re.sub(
                r"api[_-]?key[_\s]*[:=][_\s]*[a-zA-Z0-9]{20,}",
                "[API_KEY_REDACTED]",
                message,
                flags=re.IGNORECASE,
            )
            message = re.sub(
                r"token[_\s]*[:=][_\s]*[a-zA-Z0-9]{20,}",
                "[TOKEN_REDACTED]",
                message,
                flags=re.IGNORECASE,
            )

            # Remove email addresses
            message = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "[EMAIL_REDACTED]",
                message,
            )

            # Remove IP addresses
            message = re.sub(
                r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "[IP_REDACTED]", message
            )

            # Remove credit card numbers
            message = re.sub(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[CARD_REDACTED]", message)

            return message

        except Exception as e:
            logger.error(f"Error sanitizing message: {e}")
            return message

    def _build_error_context(
        self,
        context: Optional[Dict[str, Any]],
        request_id: Optional[str],
        session_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Build error context information."""
        try:
            error_context = {}

            if self.config.include_request_context and context:
                error_context.update(context)

            if self.config.include_timestamp:
                error_context["error_timestamp"] = time.time()

            if self.config.include_request_id and request_id:
                error_context["request_id"] = request_id

            if self.config.include_session_info and session_id:
                error_context["session_id"] = session_id

            return error_context if error_context else None

        except Exception as e:
            logger.error(f"Error building error context: {e}")
            return None

    def _get_suggested_actions(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
    ) -> List[str]:
        """Get suggested actions for the error."""
        try:
            suggestions = []

            if category == ErrorCategory.VALIDATION:
                suggestions.extend(
                    [
                        "Check input parameters for validity",
                        "Ensure all required fields are provided",
                        "Verify data types and formats",
                    ]
                )

            elif category == ErrorCategory.AUTHENTICATION:
                suggestions.extend(
                    [
                        "Verify API credentials are correct",
                        "Check if authentication token is valid",
                        "Ensure proper authentication headers are set",
                    ]
                )

            elif category == ErrorCategory.AUTHORIZATION:
                suggestions.extend(
                    [
                        "Check user permissions",
                        "Verify access rights for the requested resource",
                        "Contact administrator for access",
                    ]
                )

            elif category == ErrorCategory.RATE_LIMIT:
                suggestions.extend(
                    [
                        "Wait before retrying the request",
                        "Check rate limit settings",
                        "Consider implementing exponential backoff",
                    ]
                )

            elif category == ErrorCategory.TIMEOUT:
                suggestions.extend(
                    [
                        "Check network connectivity",
                        "Verify provider availability",
                        "Consider increasing timeout settings",
                    ]
                )

            elif category == ErrorCategory.PROVIDER:
                suggestions.extend(
                    [
                        "Check provider status",
                        "Verify provider configuration",
                        "Try alternative providers if available",
                    ]
                )

            elif category == ErrorCategory.NETWORK:
                suggestions.extend(
                    [
                        "Check network connection",
                        "Verify firewall settings",
                        "Check DNS resolution",
                    ]
                )

            elif category == ErrorCategory.SYSTEM:
                suggestions.extend(
                    [
                        "Contact system administrator",
                        "Check system logs for more details",
                        "Verify system resources",
                    ]
                )

            # Add general suggestions based on severity
            if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                suggestions.append("Consider implementing retry logic")
                suggestions.append("Monitor error frequency and patterns")

            return suggestions

        except Exception as e:
            logger.error(f"Error getting suggested actions: {e}")
            return ["Contact support for assistance"]

    def _create_fallback_error(
        self,
        error: Exception,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> FormattedError:
        """Create a fallback error format."""
        return FormattedError(
            error_message="An unexpected error occurred",
            error_type="UnknownError",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.UNKNOWN,
            timestamp=time.time(),
            request_id=request_id,
            session_id=session_id,
            error_code="ERR-UNK-FALLBACK",
            context={"original_error": str(error)},
        )

    def get_error_recommendations(self) -> List[str]:
        """Get recommendations for error formatting."""
        try:
            recommendations = []

            # Check if stack trace is disabled for production
            if self.config.include_stack_trace:
                recommendations.append(
                    "Consider disabling stack traces in production for security"
                )

            # Check if sensitive data sanitization is disabled
            if not self.config.sanitize_sensitive_data:
                recommendations.append("Consider enabling sensitive data sanitization")

            # Check if error categorization is disabled
            if not self.config.enable_error_categorization:
                recommendations.append(
                    "Consider enabling error categorization for better error handling"
                )

            # No recommendations
            if not recommendations:
                recommendations.append("Error formatting configuration appears optimal")

            return recommendations

        except Exception as e:
            logger.error(f"Error getting error recommendations: {e}")
            return ["Error calculating recommendations"]

    def get_error_health(self) -> Dict[str, Any]:
        """Get error formatting health information."""
        try:
            current_time = time.time()

            return {
                "status": "healthy",
                "config": self.get_error_stats(),
                "timestamp": current_time,
            }

        except Exception as e:
            logger.error(f"Error getting error health: {e}")
            return {"status": "error", "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """
        Get error formatter statistics.

        Returns:
            Dict containing error formatting statistics
        """
        return {
            "total_errors_formatted": getattr(self, "total_errors_formatted", 0),
            "errors_by_category": getattr(self, "errors_by_category", {}),
            "errors_by_severity": getattr(self, "errors_by_severity", {}),
            "formatting_errors": getattr(self, "formatting_errors", 0),
            "config": {
                "include_stack_trace": self.config.include_stack_trace,
                "include_request_context": self.config.include_request_context,
                "include_timestamp": self.config.include_timestamp,
                "include_error_code": self.config.include_error_code,
                "include_severity": self.config.include_severity,
                "include_category": self.config.include_category,
                "sanitize_sensitive_data": self.config.sanitize_sensitive_data,
                "max_error_message_length": self.config.max_error_message_length,
                "enable_error_categorization": self.config.enable_error_categorization,
            },
            "timestamp": time.time(),
        }
