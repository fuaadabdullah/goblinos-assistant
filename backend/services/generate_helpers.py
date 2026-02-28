"""Shared helpers for the generate pipeline.

Small, pure utilities used by ``generate_service`` and friends.
"""

from __future__ import annotations

import httpx


def is_retryable_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectError,
        ),
    )


def is_auth_error(exc: Exception) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response is not None
        and exc.response.status_code in {401, 403}
    )


def is_rate_limited_error(exc: Exception) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response is not None
        and exc.response.status_code == 429
    )


def safe_err(provider: str, exc: Exception) -> str:
    """Return a short, safe summary of a provider error for logging/UI."""
    try:
        if isinstance(exc, httpx.HTTPStatusError):
            return f"{provider}: HTTP {exc.response.status_code}"
        if isinstance(exc, (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout)):
            return f"{provider}: timeout"
        if isinstance(exc, httpx.ConnectError):
            return f"{provider}: connect_error"
        return f"{provider}: {type(exc).__name__}"
    except Exception:
        return f"{provider}: {type(exc).__name__}"


def is_simple_prompt(messages: list[dict[str, str]]) -> bool:
    user_messages = [m for m in messages if m.get("role") == "user"]
    if len(user_messages) != 1:
        return False
    text = (user_messages[0].get("content") or "").strip()
    return bool(text) and len(text) <= 32


def build_outage_fallback_text(prompt: str) -> str:
    """Return a canned response when all providers are down."""
    text = (prompt or "").strip().lower()
    if text in {"hi", "hello", "hey", "yo", "sup"}:
        return (
            "Hi. I'm online, but model providers are temporarily unavailable. "
            "Please try again in about a minute."
        )
    return (
        "I can't reach model providers right now. "
        "Please retry in about a minute."
    )


def normalize_provider_id(
    value: str | None,
    known_provider_ids: set[str] | None = None,
) -> str:
    """Canonicalise a provider ID string. Delegates to ``provider_catalog``."""
    try:
        from .provider_catalog import canonicalize_provider_id
    except ImportError:
        from services.provider_catalog import canonicalize_provider_id  # type: ignore[no-redef]

    return canonicalize_provider_id(value, known_provider_ids=known_provider_ids)
