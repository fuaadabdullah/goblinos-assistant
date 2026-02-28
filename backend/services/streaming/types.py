"""Types for streaming helpers."""


class StreamValidationResult:
    """Result of stream validation."""

    def __init__(self, is_valid: bool, errors: list[str]):
        self.is_valid = is_valid
        self.errors = errors


class RateLimitResult:
    """Result of rate limit check."""

    def __init__(self, allowed: bool, retry_after: float | None):
        self.allowed = allowed
        self.retry_after = retry_after
