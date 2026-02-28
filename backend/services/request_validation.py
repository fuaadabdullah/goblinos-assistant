"""
Request validation service for chat completion requests.
Handles all validation logic for chat completion parameters.
"""

from typing import List, Dict, Any, Optional, Union, TypeVar
from backend.errors import raise_validation_error
from ..schemas.v1.chat import ChatCompletionRequest, ChatMessage

# Type variables for generic validation
T = TypeVar("T")
NumericType = Union[int, float]


class ValidationError(Exception):
    """Custom exception for validation errors with structured information."""

    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"{field}: {message}")


class ValidationContext:
    """Context for collecting validation errors during request validation."""

    def __init__(self):
        self.errors: Dict[str, List[str]] = {}

    def add_error(self, field: str, message: str):
        """Add an error to the validation context."""
        if field not in self.errors:
            self.errors[field] = []
        self.errors[field].append(message)

    def has_errors(self) -> bool:
        """Check if there are any validation errors."""
        return len(self.errors) > 0

    def raise_if_errors(self, error_message: str = "Request validation failed"):
        """Raise validation error if any errors exist."""
        if self.has_errors():
            raise_validation_error(error_message, errors=self.errors)


def validate_parameter(
    value: Optional[T],
    field_name: str,
    context: ValidationContext,
    min_value: Optional[NumericType] = None,
    max_value: Optional[NumericType] = None,
    valid_values: Optional[List[str]] = None,
    default_value: Optional[T] = None,
    error_message: Optional[str] = None,
) -> T:
    """
    Generic parameter validation function.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        context: ValidationContext to collect errors
        min_value: Minimum allowed value (for numeric types)
        max_value: Maximum allowed value (for numeric types)
        valid_values: List of valid string values
        default_value: Default value to return if value is None
        error_message: Custom error message template

    Returns:
        The validated value or default value
    """
    if value is None:
        return default_value

    # Validate numeric range
    validate_numeric_range(value, field_name, context, min_value, max_value)

    # Validate against valid values list
    validate_against_valid_values(value, field_name, context, valid_values)

    return value


def validate_numeric_range(
    value: T,
    field_name: str,
    context: ValidationContext,
    min_value: Optional[NumericType],
    max_value: Optional[NumericType],
) -> None:
    """Validate numeric value against min/max range."""
    if isinstance(value, (int, float)) and (
        min_value is not None or max_value is not None
    ):
        if min_value is not None and value < min_value:
            context.add_error(
                field_name, f"Must be greater than or equal to {min_value}"
            )
        if max_value is not None and value > max_value:
            context.add_error(field_name, f"Must be less than or equal to {max_value}")


def validate_against_valid_values(
    value: T,
    field_name: str,
    context: ValidationContext,
    valid_values: Optional[List[str]],
) -> None:
    """Validate string value against list of valid values."""
    if isinstance(value, str) and valid_values is not None:
        if value not in valid_values:
            context.add_error(field_name, f"Must be one of: {', '.join(valid_values)}")


def validate_message_count(messages: List[ChatMessage], context: ValidationContext):
    """Validate message count constraints."""
    if not messages:
        context.add_error("messages", "At least one message is required")
        return

    if len(messages) > 50:
        context.add_error("messages", "Maximum 50 messages allowed")


def validate_message_content(messages: List[ChatMessage], context: ValidationContext):
    """Validate individual message content."""
    for i, msg in enumerate(messages):
        if not msg.content or not msg.content.strip():
            context.add_error("messages", f"Message {i}: content cannot be empty")

        if len(msg.content) > 10000:  # 10KB per message
            context.add_error("messages", f"Message {i}: content too long (max 10KB)")


def validate_total_content_length(
    messages: List[ChatMessage], context: ValidationContext
):
    """Validate total content length across all messages."""
    total_content_length = sum(len(msg.content) for msg in messages)

    if total_content_length > 50000:  # 50KB total
        context.add_error("messages", "Total message content too long (max 50KB)")


def validate_message_roles(messages: List[ChatMessage], context: ValidationContext):
    """Validate message roles."""
    valid_roles = ["user", "assistant", "system"]

    for i, msg in enumerate(messages):
        if msg.role not in valid_roles:
            context.add_error("messages", f"Message {i}: invalid role '{msg.role}'")


def validate_chat_request(request: ChatCompletionRequest):
    """Validate chat completion request parameters."""
    context = ValidationContext()

    # Validate messages using extracted functions
    validate_message_count(request.messages, context)
    validate_message_content(request.messages, context)
    validate_total_content_length(request.messages, context)
    validate_message_roles(request.messages, context)

    # Validate numeric parameters using generic function
    validate_parameter(
        request.temperature,
        "temperature",
        context,
        min_value=0,
        max_value=2,
        default_value=0.2,
    )

    validate_parameter(
        request.max_tokens,
        "max_tokens",
        context,
        min_value=1,
        max_value=4096,
        default_value=512,
    )

    validate_parameter(
        request.top_p, "top_p", context, min_value=0, max_value=1, default_value=0.95
    )

    # Validate string parameters using generic function
    validate_parameter(
        request.latency_target,
        "latency_target",
        context,
        valid_values=["ultra_low", "low", "medium", "high"],
        default_value="medium",
    )

    validate_parameter(
        request.intent,
        "intent",
        context,
        valid_values=[
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
            "analyze",
            "solve",
            "reason",
        ],
    )

    # Raise any collected errors
    context.raise_if_errors("Request validation failed")


def validate_temperature(temperature: Optional[float]) -> float:
    """Validate and normalize temperature parameter."""
    context = ValidationContext()
    result = validate_parameter(
        temperature, "temperature", context, min_value=0, max_value=2, default_value=0.2
    )
    context.raise_if_errors("Temperature validation failed")
    return result


def validate_max_tokens(max_tokens: Optional[int]) -> int:
    """Validate and normalize max_tokens parameter."""
    context = ValidationContext()
    result = validate_parameter(
        max_tokens,
        "max_tokens",
        context,
        min_value=1,
        max_value=4096,
        default_value=512,
    )
    context.raise_if_errors("Max tokens validation failed")
    return result


def validate_top_p(top_p: Optional[float]) -> float:
    """Validate and normalize top_p parameter."""
    context = ValidationContext()
    result = validate_parameter(
        top_p, "top_p", context, min_value=0, max_value=1, default_value=0.95
    )
    context.raise_if_errors("Top_p validation failed")
    return result


def validate_latency_target(latency_target: Optional[str]) -> str:
    """Validate and normalize latency target parameter."""
    context = ValidationContext()
    result = validate_parameter(
        latency_target,
        "latency_target",
        context,
        valid_values=["ultra_low", "low", "medium", "high"],
        default_value="medium",
    )
    context.raise_if_errors("Latency target validation failed")
    return result


def validate_intent(intent: Optional[str]) -> Optional[str]:
    """Validate intent parameter."""
    context = ValidationContext()
    valid_intents = [
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
        "analyze",
        "solve",
        "reason",
    ]
    result = validate_parameter(intent, "intent", context, valid_values=valid_intents)
    context.raise_if_errors("Intent validation failed")
    return result
