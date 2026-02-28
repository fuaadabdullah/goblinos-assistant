"""Gateway exception types shared across backend modules."""


class GatewayError(Exception):
    """Base exception for gateway-level errors."""


class TokenBudgetExceeded(GatewayError):
    """Raised when token budget is exceeded."""


class MaxTokensExceeded(GatewayError):
    """Raised when max tokens per request is exceeded."""

