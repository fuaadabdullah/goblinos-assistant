"""
RequestValidator Service for validating chat completion requests.

This service handles all request validation logic, separating it from
the main chat completion workflow for better testability and maintainability.
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class RequestValidator:
    """Service for validating chat completion requests."""

    def __init__(self):
        """Initialize the RequestValidator."""
        self.max_context_length = 100000  # Maximum context tokens
        self.max_tokens_limit = 4096  # Maximum tokens per request
        self.supported_intents = [
            "code-gen",
            "creative",
            "explain",
            "summarize",
            "rag",
            "retrieval",
            "chat",
            "classification",
            "status",
            "translation",
        ]

    def validate_chat_request(self, request: Any) -> None:
        """
        Validate a chat completion request.

        Args:
            request: The chat completion request to validate

        Raises:
            HTTPException: If validation fails
        """
        try:
            # Validate messages
            self._validate_messages(request.messages)

            # Validate model (optional)
            if hasattr(request, 'model') and request.model:
                self._validate_model(request.model)

            # Validate max_tokens (optional)
            if hasattr(request, 'max_tokens') and request.max_tokens:
                self._validate_max_tokens(request.max_tokens)

            # Validate intent (optional)
            if hasattr(request, 'intent') and request.intent:
                self._validate_intent(request.intent)

            # Validate context (optional)
            if hasattr(request, 'context') and request.context:
                self._validate_context(request.context)

            # Validate temperature (optional)
            if hasattr(request, 'temperature') and request.temperature is not None:
                self._validate_temperature(request.temperature)

            # Validate top_p (optional)
            if hasattr(request, 'top_p') and request.top_p is not None:
                self._validate_top_p(request.top_p)

            logger.debug(f"Request validation passed for request")

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Request validation failed: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request format: {str(e)}"
            )

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate the messages array."""
        if not messages:
            raise HTTPException(
                status_code=400,
                detail="Messages array cannot be empty"
            )

        if not isinstance(messages, list):
            raise HTTPException(
                status_code=400,
                detail="Messages must be an array"
            )

        # Check message structure
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"Message {i} must be an object"
                )

            if "role" not in message or "content" not in message:
                raise HTTPException(
                    status_code=400,
                    detail=f"Message {i} must have 'role' and 'content' fields"
                )

            if message["role"] not in ["user", "assistant", "system"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Message {i} has invalid role: {message['role']}"
                )

            if not isinstance(message["content"], str):
                raise HTTPException(
                    status_code=400,
                    detail=f"Message {i} content must be a string"
                )

            if not message["content"].strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"Message {i} content cannot be empty"
                )

    def _validate_model(self, model: str) -> None:
        """Validate the model parameter."""
        if not isinstance(model, str):
            raise HTTPException(
                status_code=400,
                detail="Model must be a string"
            )

        if not model.strip():
            raise HTTPException(
                status_code=400,
                detail="Model cannot be empty"
            )

        # Note: We don't validate against specific model names here
        # as models may be dynamic and provider-specific

    def _validate_max_tokens(self, max_tokens: int) -> None:
        """Validate the max_tokens parameter."""
        if not isinstance(max_tokens, int):
            raise HTTPException(
                status_code=400,
                detail="max_tokens must be an integer"
            )

        if max_tokens <= 0:
            raise HTTPException(
                status_code=400,
                detail="max_tokens must be greater than 0"
            )

        if max_tokens > self.max_tokens_limit:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens cannot exceed {self.max_tokens_limit}"
            )

    def _validate_intent(self, intent: str) -> None:
        """Validate the intent parameter."""
        if not isinstance(intent, str):
            raise HTTPException(
                status_code=400,
                detail="Intent must be a string"
            )

        if intent not in self.supported_intents:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported intent: {intent}. Supported intents: {', '.join(self.supported_intents)}"
            )

    def _validate_context(self, context: str) -> None:
        """Validate the context parameter."""
        if not isinstance(context, str):
            raise HTTPException(
                status_code=400,
                detail="Context must be a string"
            )

        # Estimate token count (rough approximation: 4 chars per token)
        estimated_tokens = len(context) // 4
        if estimated_tokens > self.max_context_length:
            raise HTTPException(
                status_code=400,
                detail=f"Context too long. Estimated {estimated_tokens} tokens, maximum allowed: {self.max_context_length}"
            )

    def _validate_temperature(self, temperature: float) -> None:
        """Validate the temperature parameter."""
        if not isinstance(temperature, (int, float)):
            raise HTTPException(
                status_code=400,
                detail="Temperature must be a number"
            )

        if not 0.0 <= temperature <= 2.0:
            raise HTTPException(
                status_code=400,
                detail="Temperature must be between 0.0 and 2.0"
            )

    def _validate_top_p(self, top_p: float) -> None:
        """Validate the top_p parameter."""
        if not isinstance(top_p, (int, float)):
            raise HTTPException(
                status_code=400,
                detail="top_p must be a number"
            )

        if not 0.0 <= top_p <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="top_p must be between 0.0 and 1.0"
            )

    def validate_user_input(self, user_message: str) -> bool:
        """
        Validate user input for content safety and appropriateness.

        Args:
            user_message: The user's message to validate

        Returns:
            bool: True if input is valid, False otherwise
        """
        if not user_message or not user_message.strip():
            return False

        # Basic profanity check (simple implementation)
        profanity_list = ["badword1", "badword2"]  # Extend as needed
        user_lower = user_message.lower()
        for word in profanity_list:
            if word in user_lower:
                logger.warning(f"Potentially inappropriate content detected in message")
                return False

        # Check for excessively long messages
        if len(user_message) > 10000:  # 10k characters
            logger.warning(f"Message too long: {len(user_message)} characters")
            return False

        return True

    def sanitize_input(self, user_message: str) -> str:
        """
        Sanitize user input to remove potentially harmful content.

        Args:
            user_message: The user's message to sanitize

        Returns:
            str: Sanitized message
        """
        if not user_message:
            return ""

        # Remove excessive whitespace
        sanitized = " ".join(user_message.split())

        # Remove potentially harmful characters (basic implementation)
        # In a real implementation, you might want more sophisticated sanitization
        dangerous_chars = ["<", ">", "&", "\"", "'"]
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")

        return sanitized.strip()