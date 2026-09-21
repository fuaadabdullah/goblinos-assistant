"""FreeSerp keyless web-search client.

FreeSerp (https://freeserp.ai) is a web search API with no key, no signup,
no account and no quota: one GET request returns ranked JSON over an
independent web index, with an LLM-written summary attached to every
result. That makes it the zero-setup fallback for chat web-search
enrichment — it works even when BRAVE_API_KEY is not set.

Chat integration mirrors :mod:`backend.services.brave_search`:
:func:`should_search` decides whether the latest user message warrants a
live lookup, and :func:`format_results_for_llm` turns the results into a
system-message block the model can ground on. See ``chat_context.py``,
which prefers Brave when configured and falls back to FreeSerp.

Optional courtesy identity (per FreeSerp's docs, everything is optional
and nothing is validated): ``FREESERP_AGENT``, ``FREESERP_EMAIL``,
``FREESERP_PROJECT``, ``FREESERP_WEBSITE``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

FREESERP_ENDPOINT = "https://freeserp.ai/api.php"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_RESULT_COUNT = 5
MAX_RESULT_COUNT = 10

_AGENT_DEFAULT = "goblin-assistant"


class FreeSerpError(RuntimeError):
    """Raised for unrecoverable FreeSerp failures (unused: web_search degrades)."""


@dataclass
class FreeSerpResult:
    domain: str
    url: str
    title: str
    ai_summary: str = ""
    category: Optional[str] = None
    ai_categories: list[str] = field(default_factory=list)
    dr: Optional[int] = None
    went_live: Optional[str] = None
    real_site: Optional[int] = None


def is_configured() -> bool:
    """FreeSerp needs no key, so it is always usable."""
    return True


def web_search_enabled() -> bool:
    """Kill-switch for chat search enrichment (default on).

    Shares ``CHAT_WEB_SEARCH_ENABLED`` with the Brave integration so one
    switch governs all chat web-search enrichment.
    """
    return os.getenv("CHAT_WEB_SEARCH_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def provider_name() -> str:
    """Preferred chat web-search provider: 'brave' when it has a key, else FreeSerp."""
    override = os.getenv("CHAT_WEB_SEARCH_PROVIDER", "auto").strip().lower()
    if override in ("brave", "freeserp"):
        return override
    if override != "auto":
        logger.warning("Unknown CHAT_WEB_SEARCH_PROVIDER=%r; using auto", override)
    # Imported lazily to avoid a hard dependency cycle at import time.
    from . import brave_search

    return "brave" if brave_search.is_configured() else "freeserp"


def _courtesy_params() -> dict[str, str]:
    """Optional self-identification params (FreeSerp's documented courtesy)."""
    params: dict[str, str] = {"agent": os.getenv("FREESERP_AGENT", _AGENT_DEFAULT).strip() or _AGENT_DEFAULT}
    for key, env in (
        ("email", "FREESERP_EMAIL"),
        ("project", "FREESERP_PROJECT"),
        ("website", "FREESERP_WEBSITE"),
    ):
        value = os.getenv(env, "").strip()
        if value:
            params[key] = value
    return params


def _parse_results(data: Any, limit: int) -> list[FreeSerpResult]:
    """Parse a FreeSerp JSON payload into normalized results."""
    if not isinstance(data, dict) or data.get("ok") is not True:
        return []
    raw_results = data.get("results") or []
    results: list[FreeSerpResult] = []
    for item in raw_results:
        if len(results) >= limit:
            break
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            FreeSerpResult(
                domain=(item.get("domain") or "").strip(),
                url=url,
                title=(item.get("title") or url).strip(),
                ai_summary=(item.get("ai_summary") or "").strip(),
                category=item.get("category"),
                ai_categories=list(item.get("ai_categories") or []),
                dr=item.get("dr"),
                went_live=item.get("went_live"),
                real_site=item.get("real_site"),
            )
        )
    return results


async def web_search(
    query: str,
    *,
    count: int = DEFAULT_RESULT_COUNT,
    ai_startups: bool = False,
    category: Optional[str] = None,
    sort: str = "relevance",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: Any = None,
) -> list[FreeSerpResult]:
    """Run a FreeSerp web search and return normalized results.

    No key is ever needed. Transport failures return an empty list so
    callers can degrade gracefully.
    """
    count = max(1, min(int(count or DEFAULT_RESULT_COUNT), MAX_RESULT_COUNT))
    params: dict[str, Any] = {
        "q": query,
        "size": count,
        "sort": sort,
        **_courtesy_params(),
    }
    if ai_startups:
        params["ai_startups"] = 1
    if category:
        params["category"] = category

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)
        close_client = True
    try:
        response = await client.get(FREESERP_ENDPOINT, params=params)
        response.raise_for_status()
        return _parse_results(response.json(), count)
    except Exception:
        logger.warning("FreeSerp web search failed for query %r", query, exc_info=True)
        return []
    finally:
        if close_client:
            await client.aclose()


async def search_ai_startups(
    *,
    ai_category: Optional[str] = None,
    from_date: Optional[str] = None,
    count: int = DEFAULT_RESULT_COUNT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: Any = None,
) -> list[FreeSerpResult]:
    """Discover recently-live AI startups (``ai_startups=1`` filter).

    Browse mode: no query text needed. ``from_date`` is ``YYYY-MM-DD`` and
    results come back in launch order (``went_live`` descending).
    """
    count = max(1, min(int(count or DEFAULT_RESULT_COUNT), MAX_RESULT_COUNT))
    params: dict[str, Any] = {
        "ai_startups": 1,
        "sort": "went_live",
        "order": "desc",
        "size": count,
        **_courtesy_params(),
    }
    if ai_category:
        params["ai_categories"] = ai_category
    if from_date:
        params["from_date"] = from_date

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)
        close_client = True
    try:
        response = await client.get(FREESERP_ENDPOINT, params=params)
        response.raise_for_status()
        return _parse_results(response.json(), count)
    except Exception:
        logger.warning("FreeSerp AI-startup discovery failed", exc_info=True)
        return []
    finally:
        if close_client:
            await client.aclose()


def format_results_for_llm(query: str, results: list[FreeSerpResult]) -> str:
    """Format search results as a system-message block for the chat model."""
    lines = [
        f'Web search results for "{query}" (via FreeSerp). '
        "Use them to answer questions about current or external facts, "
        "and cite sources inline as [1], [2], etc."
    ]
    for i, result in enumerate(results, 1):
        block = f"[{i}] {result.title}\n{result.url}"
        meta: list[str] = []
        if result.dr is not None:
            meta.append(f"DR {result.dr}")
        if result.went_live:
            meta.append(f"live since {result.went_live}")
        if meta:
            block += f"\n({', '.join(meta)})"
        if result.ai_summary:
            block += f"\n{result.ai_summary}"
        lines.append(block)
    return "\n\n".join(lines)


# --- Chat-search trigger heuristic -------------------------------------------
# Kept in parity with brave_search: only fires on an explicit search
# request or a strong recency signal. Unlike Brave, no API key is required.

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
    """Return the query to search, or None when chat should answer directly."""
    if not web_search_enabled():
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
    "FREESERP_ENDPOINT",
    "FreeSerpError",
    "FreeSerpResult",
    "format_results_for_llm",
    "is_configured",
    "provider_name",
    "search_ai_startups",
    "should_search",
    "web_search",
    "web_search_enabled",
]
