"""
Base provider interface for unified provider adapters.

All providers (cloud + local) implement this interface for consistent routing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class InferenceRequest:
    """Standardized inference request."""

    messages: list[Dict[str, str]]
    model: str
    model_family: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False


@dataclass
class InferenceResult:
    """Standardized inference result."""

    content: str
    usage: Dict[str, int]  # input_tokens, output_tokens, total_tokens
    model: str
    finish_reason: Optional[str] = None
    latency_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None


class ProviderBase(ABC):
    """Unified interface for all AI providers (cloud + local)."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """Provider capabilities including supported models, token limits, etc.

        Returns:
            Dict with keys:
            - models: List[str] - supported model names
            - max_tokens: Dict[str, int] - max tokens per model
            - supports_streaming: bool
            - supports_vision: bool
            - cost_per_token_input: float
            - cost_per_token_output: float
        """
        pass

    @abstractmethod
    def health_check(self) -> HealthStatus:
        """Check provider health and availability.

        Returns:
            HealthStatus indicating provider operational state
        """
        pass

    @abstractmethod
    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute inference request.

        Args:
            request: Standardized inference request

        Returns:
            InferenceResult with response data and metadata
        """
        pass

    @abstractmethod
    def estimate_cost(self, request: InferenceRequest) -> float:
        """Estimate cost for the given request in USD.

        Args:
            request: Inference request to estimate cost for

        Returns:
            Estimated cost in USD
        """
        pass


# Helper functions for retries, backoff, and normalization
def with_retry(
    func, max_retries: int = 3, backoff_factor: float = 2.0, base_delay: float = 1.0
):
    """Decorator to add retry logic with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier
        base_delay: Base delay in seconds

    Returns:
        Function result or raises last exception
    """

    def wrapper(*args, **kwargs):
        import time
        import random

        last_exception = None
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    raise e

                # Exponential backoff with jitter
                delay = base_delay * (backoff_factor**attempt) + random.uniform(0, 1)
                time.sleep(delay)

        # This should never be reached
        raise last_exception

    return wrapper


def normalize_messages(messages: list[Dict[str, str]]) -> list[Dict[str, str]]:
    """Normalize message format across providers.

    Ensures all messages have 'role' and 'content' keys.
    Handles various input formats and converts to standard format.
    """
    normalized = []
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            normalized.append(msg)
        elif isinstance(msg, dict) and "content" in msg:
            # Assume user role if not specified
            normalized.append({"role": "user", "content": msg["content"]})
        elif isinstance(msg, str):
            # Convert string messages to user role
            normalized.append({"role": "user", "content": msg})
        else:
            # Convert other types to string
            normalized.append({"role": "user", "content": str(msg)})

    return normalized


def emit_metrics(
    provider: str, model: str, latency_ms: int, success: bool, **extra_labels
):
    """Emit standardized metrics for all providers.

    Args:
        provider: Provider identifier
        model: Model name used
        latency_ms: Response latency in milliseconds
        success: Whether the request was successful
        **extra_labels: Additional metric labels
    """
    import logging

    logger = logging.getLogger("providers.metrics")

    labels = {
        "provider": provider,
        "model": model,
        "latency_ms": latency_ms,
        "success": success,
        **extra_labels,
    }

    logger.info("Provider inference metrics", extra=labels)
