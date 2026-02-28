"""
Helper functions for ChatHandler to reduce cognitive complexity.

This module contains utility functions extracted from ChatHandler
to improve maintainability and reduce method complexity.
"""

import logging
from typing import Any, Dict

from .utils import get_provider_class

logger = logging.getLogger(__name__)


def resolve_provider_instance(provider_config: Dict[str, Any]) -> Any:
    """
    Resolve and instantiate a provider from configuration.

    Args:
        provider_config: Provider configuration dictionary

    Returns:
        Instantiated provider instance

    Raises:
        ValueError: If provider cannot be resolved
    """
    try:
        provider_name = provider_config["provider_name"]
        provider_class = get_provider_class(provider_name)
        provider_instance = provider_class(provider_config)

        logger.debug(f"Resolved provider: {provider_name}")
        return provider_instance

    except KeyError as e:
        logger.error(f"Missing provider configuration key: {e}")
        raise ValueError(f"Invalid provider configuration: missing {e}")
    except Exception as e:
        logger.error(
            f"Failed to resolve provider {provider_config.get('provider_name', 'unknown')}: {e}"
        )
        raise ValueError(f"Provider resolution failed: {e}")


async def validate_and_check_limits(
    validator,
    rate_limiter,
    error_formatter,
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    user_id: str = None,
    client_ip: str = None,
    session_id: str = None,
):
    """
    Validate chat request and check rate limits.

    Args:
        validator: Chat validator instance
        rate_limiter: Rate limiter instance
        error_formatter: Error formatter instance
        messages: Chat messages
        model: Model name
        temperature: Temperature setting
        max_tokens: Max tokens
        stream: Whether streaming
        user_id: User ID for rate limiting
        client_ip: Client IP for rate limiting
        session_id: Session ID for rate limiting

    Returns:
        None

    Raises:
        HTTPException: If validation fails or rate limit exceeded
    """
    from fastapi import HTTPException

    # Validate request
    validation_result = await validator.validate_chat_request(
        messages, model, temperature, max_tokens, stream
    )
    if not validation_result.is_valid:
        raise HTTPException(
            status_code=400,
            detail=error_formatter.format_validation_error(validation_result.errors),
        )

    # Check rate limits
    rate_limit_result = await rate_limiter.check_rate_limit(
        user_id, client_ip, session_id
    )
    if not rate_limit_result.allowed:
        raise HTTPException(
            status_code=429,
            detail=error_formatter.format_rate_limit_error(
                rate_limit_result.retry_after
            ),
        )
