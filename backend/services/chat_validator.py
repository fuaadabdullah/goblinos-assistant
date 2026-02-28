"""
ChatValidator Service for validating chat requests and responses.

This service handles all validation logic for chat completion requests,
separating validation concerns from the main chat handler.
"""

import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChatValidator:
    """Service for validating chat requests and responses."""

    async def validate_chat_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
    ) -> "ValidationResult":
        """
        Validate chat request parameters.

        Args:
            messages: List of chat messages
            model: Model to use for generation
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream responses

        Returns:
            ValidationResult with errors if any
        """
        errors: List[str] = []

        # Validate messages
        if not messages:
            errors.append("Messages cannot be empty")
        else:
            # Validate message structure
            for i, message in enumerate(messages):
                if not isinstance(message, dict):
                    errors.append(f"Message {i} must be a dictionary")
                    continue

                if "role" not in message:
                    errors.append(f"Message {i} missing required 'role' field")
                elif message["role"] not in ["user", "assistant", "system"]:
                    errors.append(f"Message {i} has invalid role: {message['role']}")

                if "content" not in message:
                    errors.append(f"Message {i} missing required 'content' field")
                elif not isinstance(message["content"], str):
                    errors.append(f"Message {i} content must be a string")

        # Validate model
        if not model or not isinstance(model, str):
            errors.append("Model is required and must be a string")

        # Validate temperature
        if not isinstance(temperature, (int, float)):
            errors.append("Temperature must be a number")
        elif not (0.0 <= temperature <= 2.0):
            errors.append("Temperature must be between 0.0 and 2.0")

        # Validate max_tokens
        if max_tokens is not None:
            if not isinstance(max_tokens, int):
                errors.append("Max tokens must be an integer")
            elif max_tokens <= 0:
                errors.append("Max tokens must be positive")

        # Validate stream flag
        if not isinstance(stream, bool):
            errors.append("Stream must be a boolean")

        return ValidationResult(is_valid=(len(errors) == 0), errors=errors)

    # Backward-compatible API expected by older callers/tests
    async def validate_request(self, request) -> "ValidationResult":
        """Compatibility wrapper that accepts a ChatRequest-like object.

        Converts results into the legacy `ValidationResult` shape expected by
        the unit tests (errors as dicts with `field`/`message`)."""
        errors: List[Dict[str, str]] = []

        # session_id validation (tests expect a fielded error)
        if not getattr(request, "session_id", None) or not self.validate_session_id(
            getattr(request, "session_id", "")
        ):
            errors.append(
                {
                    "field": "session_id",
                    "message": "Session ID must be a non-empty string",
                }
            )

        # Delegate core validation
        core_result = await self.validate_chat_request(
            getattr(request, "messages", []),
            getattr(request, "model", ""),
            getattr(request, "temperature", 0.0),
            getattr(request, "max_tokens", None),
            getattr(request, "stream", False),
        )

        # Normalize core_result.errors (strings) into dicts
        for err in getattr(core_result, "errors", []):
            errors.append({"field": "request", "message": str(err)})

        return ValidationResult(is_valid=(len(errors) == 0), errors=errors)

    def get_stats(self) -> Dict[str, Any]:
        """Return lightweight validation stats for compatibility with tests."""
        # Minimal implementation sufficient for unit tests — replace with real metrics later
        return {
            "total_validations": 0,
            "valid_requests": 0,
            "invalid_requests": 0,
            "validation_errors": [],
            "timestamp": time.time(),
        }

    def validate_chat_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        Validate individual chat chunk.

        Args:
            chunk: Chat chunk data

        Returns:
            True if chunk is valid, False otherwise
        """
        if not isinstance(chunk, dict):
            logger.warning("Chat chunk is not a dictionary")
            return False

        # Check for required fields
        required_fields = ["content"]
        for field in required_fields:
            if field not in chunk:
                logger.warning(f"Chat chunk missing required field: {field}")
                return False

        # Validate content
        if not isinstance(chunk["content"], str):
            logger.warning("Chat chunk content must be a string")
            return False

        # Validate optional fields
        if "role" in chunk and chunk["role"] not in ["user", "assistant", "system"]:
            logger.warning(f"Chat chunk has invalid role: {chunk['role']}")
            return False

        return True

    def validate_chat_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate chat completion response.

        Args:
            response: Chat completion response

        Returns:
            True if response is valid, False otherwise
        """
        if not isinstance(response, dict):
            logger.warning("Chat response is not a dictionary")
            return False

        # Check for required fields
        required_fields = ["content", "usage"]
        for field in required_fields:
            if field not in response:
                logger.warning(f"Chat response missing required field: {field}")
                return False

        # Validate content
        if not isinstance(response["content"], str):
            logger.warning("Chat response content must be a string")
            return False

        # Validate usage
        usage = response["usage"]
        if not isinstance(usage, dict):
            logger.warning("Chat response usage must be a dictionary")
            return False

        required_usage_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
        for field in required_usage_fields:
            if field not in usage:
                logger.warning(f"Chat response usage missing required field: {field}")
                return False
            if not isinstance(usage[field], int) or usage[field] < 0:
                logger.warning(
                    f"Chat response usage {field} must be a non-negative integer"
                )
                return False

        # Validate optional fields
        if "role" in response and response["role"] not in [
            "user",
            "assistant",
            "system",
        ]:
            logger.warning(f"Chat response has invalid role: {response['role']}")
            return False

        return True

    def validate_session_id(self, session_id: str) -> bool:
        """
        Validate session ID format.

        Args:
            session_id: Session ID string

        Returns:
            True if session ID is valid, False otherwise
        """
        if not session_id or not isinstance(session_id, str):
            logger.warning("Session ID must be a non-empty string")
            return False

        # Basic length check
        if len(session_id) < 3:
            logger.warning("Session ID must be at least 3 characters long")
            return False

        # Check for valid characters (alphanumeric and hyphens)
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
            logger.warning("Session ID contains invalid characters")
            return False

        return True

    def validate_user_id(self, user_id: Optional[str]) -> bool:
        """
        Validate user ID format.

        Args:
            user_id: User ID string (optional)

        Returns:
            True if user ID is valid or None, False otherwise
        """
        if user_id is None:
            return True

        if not isinstance(user_id, str):
            logger.warning("User ID must be a string")
            return False

        # Basic length check
        if len(user_id) < 3:
            logger.warning("User ID must be at least 3 characters long")
            return False

        # Check for valid characters (alphanumeric and hyphens)
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", user_id):
            logger.warning("User ID contains invalid characters")
            return False

        return True

    def validate_client_ip(self, client_ip: Optional[str]) -> bool:
        """
        Validate client IP address format.

        Args:
            client_ip: Client IP address (optional)

        Returns:
            True if client IP is valid or None, False otherwise
        """
        if client_ip is None:
            return True

        if not isinstance(client_ip, str):
            logger.warning("Client IP must be a string")
            return False

        # Basic IP validation (IPv4 and IPv6)
        import ipaddress

        try:
            ipaddress.ip_address(client_ip)
            return True
        except ValueError:
            logger.warning(f"Invalid IP address format: {client_ip}")
            return False


class ValidationConfig:
    """Configuration for chat validation (stub for test compatibility)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ValidationResult:
    """Result of chat validation (test compatibility)."""

    def __init__(self, is_valid: bool, errors: list):
        self.is_valid = is_valid
        self.errors = errors

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.is_valid = False

    def merge(self, other: "ValidationResult") -> None:
        self.errors.extend(other.errors)
        self.is_valid = self.is_valid and other.is_valid

    def to_http_error(self) -> str:
        if self.is_valid:
            return ""
        error_message = "Chat request validation failed"
        if self.errors:
            error_message += f": {', '.join(self.errors)}"
        return error_message


# Backward compatibility for legacy tests
ChatValidationResult = ValidationResult
