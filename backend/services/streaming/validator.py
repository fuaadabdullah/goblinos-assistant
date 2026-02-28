"""Validation logic for streaming requests and chunks."""

from typing import Any

from .types import StreamValidationResult


class StreamValidator:
    """Validator for streaming requests and responses."""

    async def validate_stream_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> StreamValidationResult:
        """Validate streaming request parameters."""
        errors = []

        # Validate messages
        if not messages:
            errors.append("Messages cannot be empty")

        # Validate model
        if not model:
            errors.append("Model is required")

        # Validate temperature
        if not (0.0 <= temperature <= 2.0):
            errors.append("Temperature must be between 0.0 and 2.0")

        # Validate max_tokens
        if max_tokens is not None and max_tokens <= 0:
            errors.append("Max tokens must be positive")

        # Validate stream flag
        if not stream:
            errors.append("Stream must be True for streaming requests")

        return StreamValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )

    def validate_stream_chunk(self, chunk: dict[str, Any]) -> bool:
        """Validate individual stream chunk."""
        required_fields = ["content"]
        return all(field in chunk for field in required_fields)
