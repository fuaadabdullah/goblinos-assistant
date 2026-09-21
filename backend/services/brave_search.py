"""Brave Search web-search client.

Server-side only: the API key lives in the ``BRAVE_API_KEY`` environment
variable and is sent as the ``X-Subscription-Token`` header. Nothing in this
module ever exposes the key to clients.

Chat integration: :func:`should_search` decides whether the latest user
message warrants a live web lookup; :func:`format_results_for_llm` turns the
results into a system-message block the model can ground on.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_RESULT_COUNT = 5
MAX_RESULT_COUNT = 10


class BraveSearchNotConfigured(RuntimeError):
    """Raised when BRAVE_API_KEY is not set."""


@dataclass
class BraveSearchResult:
    title: str
    url: str
    snippet: str
    published: Optional[str] = None


def get_api_key() -> Optional[str]:
    """Return the configured Brave API key, or None when unset."""
    key = os.getenv("BRAVE_API_KEY", "").strip()
    return key or None


def is_configured() -> bool:
    return get_api_key() is not None


def web_search_enabled() -> bool:
    """Kill-switch for chat search enrichment (default on)."""
    return os.getenv("CHAT_WEB_SEARCH_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _parse_results(data: Any, limit: int) -> list[BraveSearchResult]:
    """Parse a Brave web-search JSON payload into normalized results."""
    if not isinstance(data, dict):
        return []
    web = data.get("web") or {}
    raw_results = web.get("results") or []
    results: list[BraveSearchResult] = []
    for item in raw_results:
        if len(results) >= limit:
            break
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            BraveSearchResult(
                title=(item.get("title") or url).strip(),
                url=url,
                snippet=(item.get("description") or "").strip(),
                published=item.get("page_age") or item.get("age"),
            )
        )
    return results


async def web_search(
    query: str,
    *,
    count: int = DEFAULT_RESULT_COUNT,
    country: str = "US",
    search_lang: str = "en",
    freshness: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    api_key: Optional[str] = None,
    client: Any = None,
) -> list[BraveSearchResult]:
    """Run a Brave web search and return normalized results.

    Raises :class:`BraveSearchNotConfigured` when no API key is available.
    Transport failures return an empty list so callers can degrade gracefully.
    """
    key = api_key or get_api_key()
    if not key:
        raise BraveSearchNotConfigured("BRAVE_API_KEY is not set")

    count = max(1, min(int(count or DEFAULT_RESULT_COUNT), MAX_RESULT_COUNT))
    params: dict[str, Any] = {
        "q": query,
        "count": count,
        "country": country,
        "search_lang": search_lang,
        "safesearch": "moderate",
    }
    if freshness:
        params["freshness"] = freshness
    headers = {"Accept": "application/json", "X-Subscription-Token": key}

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)
        close_client = True
    try:
        response = await client.get(BRAVE_SEARCH_ENDPOINT, params=params, headers=headers)
        response.raise_for_status()
        return _parse_results(response.json(), count)
    except Exception:
        logger.warning("Brave web search failed for query %r", query, exc_info=True)
        return []
    finally:
        if close_client:
            await client.aclose()


def format_results_for_llm(query: str, results: list[BraveSearchResult]) -> str:
    """Format search results as a system-message block for the chat model."""
    lines = [
        f'Web search results for "{query}" (via Brave Search). '
        "Use them to answer questions about current or external facts, "
        "and cite sources inline as [1], [2], etc."
    ]
    for i, result in enumerate(results, 1):
        block = f"[{i}] {result.title}\n{result.url}"
        if result.snippet:
            block += f"\n{result.snippet}"
        lines.append(block)
    return "\n\n".join(lines)


# --- Chat-search trigger heuristic -------------------------------------------

_EXPLICIT_SEARCH_RE = re.compile(
    r"\b(search the web|look (it|this|that) up|google it|web search|search online)\b",
    re.IGNORECASE,
)
_RECENCY_RE = re.compile(
    r"\b(latest|newest|current|today|yesterday|last night|this (week|morning|month|year)"
    r"|right now|breaking|news|2026)\b",
    re.IGNORECASE,
)
_MAX_QUERY_CHARS = 500


def should_search(messages: list[dict[str, Any]]) -> Optional[str]:
    """Return the query to search, or None when chat should answer directly.

    Conservative on purpose: only fires on an explicit search request or a
    strong recency signal, so ordinary chit-chat never pays search latency.
    """
    if not web_search_enabled() or not is_configured():
        return None
    query: Optional[str] = None
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user" and (msg.get("content") or "").strip():
            query = msg["content"].strip()
            break
    if not query or len(query) > _MAX_QUERY_CHARS:
        return None
    if _EXPLICIT_SEARCH_RE.search(query) or _RECENCY_RE.search(query):
        return query
    return None


__all__ = [
    "BRAVE_SEARCH_ENDPOINT",
    "BraveSearchNotConfigured",
    "BraveSearchResult",
    "format_results_for_llm",
    "get_api_key",
    "is_configured",
    "should_search",
    "web_search",
    "web_search_enabled",
]
