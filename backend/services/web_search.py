"""Unified web-search facade over Brave Search and FreeSerp.

Both providers are useful for different questions, so the facade routes
each query to the right one(s) instead of treating FreeSerp as a mere
fallback:

- Startup/competitor discovery ("ai voice agent startups", "alternatives
  to X", "who else is in this space") -> FreeSerp with ``ai_startups=1``,
  newest launches first. This is the query class Brave is weakest at.
- Breaking/recency questions ("latest iPhone news") -> Brave with a
  freshness window, because Brave answers "what does Google rank now".
- Everything else -> both in parallel, merged and deduped by domain.

Either provider failing (missing Brave key, transport error) never breaks
the other or the caller: failures degrade to that provider's results being
absent. The chat kill switch ``CHAT_WEB_SEARCH_ENABLED`` still governs
everything, and ``CHAT_WEB_SEARCH_PROVIDER`` (auto|brave|freeserp|both)
can override routing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

from . import brave_search, freeserp_search
from .brave_search import BraveSearchNotConfigured

logger = logging.getLogger(__name__)

DEFAULT_RESULT_COUNT = 5
MAX_RESULT_COUNT = 10


@dataclass
class UnifiedSearchResult:
    title: str
    url: str
    summary: str = ""
    provider: str = ""  # "brave" | "freeserp"
    domain: str = ""
    published: Optional[str] = None  # Brave page_age / FreeSerp went_live
    authority: Optional[int] = None  # FreeSerp DR; Brave has none


def web_search_enabled() -> bool:
    """Kill-switch for chat search enrichment (default on)."""
    return os.getenv("CHAT_WEB_SEARCH_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def provider_override() -> str:
    """Explicit provider selection from env, or 'auto'."""
    value = os.getenv("CHAT_WEB_SEARCH_PROVIDER", "auto").strip().lower()
    if value in ("brave", "freeserp", "both", "auto"):
        return value
    logger.warning("Unknown CHAT_WEB_SEARCH_PROVIDER=%r; using auto", value)
    return "auto"


# --- Query-intent detection ---------------------------------------------------

# Queries that are really "who exists in this space" — FreeSerp's home turf:
# its ai_startups filter, launch-date sorting and LLM summaries beat a
# relevance-ranked snippet list for discovery work.
_DISCOVERY_RE = re.compile(
    r"\b(startups?|competitors?|alternatives?|new (ai |llm )?compan(y|ies)"
    r"|launched|launching|directory of|players in|landscape of|market map"
    r"|who (else|also)|emerging|upcoming)\b",
    re.IGNORECASE,
)

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


def detect_intent(query: str) -> str:
    """Classify a search query: 'discovery', 'recency', or 'general'."""
    if _DISCOVERY_RE.search(query):
        return "discovery"
    if _RECENCY_RE.search(query):
        return "recency"
    return "general"


def _freshness_for(query: str) -> Optional[str]:
    """Map a recency-flavoured query onto Brave's freshness window."""
    q = query.lower()
    if re.search(r"\b(today|last night|right now|breaking)\b", q):
        return "pd"
    if re.search(r"\b(this week|yesterday)\b", q):
        return "pw"
    if re.search(r"\b(this month|news)\b", q):
        return "pm"
    return "py"


def select_providers(query: str, override: Optional[str] = None) -> list[str]:
    """Decide which providers serve a query ('auto' routing).

    ``override``: 'auto', 'brave', 'freeserp', or 'both' (defaults to the
    ``CHAT_WEB_SEARCH_PROVIDER`` env var). Returns a list like ['brave'],
    ['freeserp'], or ['brave', 'freeserp']. Brave is skipped silently when
    its key is missing.
    """
    choice = (override or provider_override()).strip().lower()
    if choice not in ("brave", "freeserp", "both", "auto"):
        logger.warning("Unknown provider selection %r; using auto", choice)
        choice = "auto"
    if choice != "auto":
        candidates = ["brave", "freeserp"] if choice == "both" else [choice]
    else:
        intent = detect_intent(query)
        if intent == "discovery":
            candidates = ["freeserp"]
        elif intent == "recency":
            candidates = ["brave"]
        else:
            candidates = ["brave", "freeserp"]
    return [p for p in candidates if p == "freeserp" or brave_search.is_configured()]


# --- Provider adapters --------------------------------------------------------

def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _from_brave(results: list[Any]) -> list[UnifiedSearchResult]:
    out: list[UnifiedSearchResult] = []
    for r in results:
        out.append(
            UnifiedSearchResult(
                title=r.title,
                url=r.url,
                summary=r.snippet,
                provider="brave",
                domain=_domain_of(r.url),
                published=r.published,
            )
        )
    return out


def _from_freeserp(results: list[Any]) -> list[UnifiedSearchResult]:
    out: list[UnifiedSearchResult] = []
    for r in results:
        out.append(
            UnifiedSearchResult(
                title=r.title,
                url=r.url,
                summary=r.ai_summary,
                provider="freeserp",
                domain=r.domain.lower() or _domain_of(r.url),
                published=r.went_live,
                authority=r.dr,
            )
        )
    return out


async def _search_brave(query: str, count: int, timeout: float) -> list[UnifiedSearchResult]:
    try:
        results = await brave_search.web_search(
            query, count=count, freshness=_freshness_for(query), timeout=timeout
        )
        return _from_brave(results)
    except BraveSearchNotConfigured:
        logger.info("Brave Search not configured; skipping provider")
        return []
    except Exception:
        logger.warning("Brave provider failed for query %r", query, exc_info=True)
        return []


async def _search_freeserp(
    query: str, count: int, timeout: float, ai_startups: bool
) -> list[UnifiedSearchResult]:
    try:
        if ai_startups:
            results = await freeserp_search.search_ai_startups(
                count=count, timeout=timeout
            )
            if not results:
                # Fall back to a plain relevance search before giving up.
                results = await freeserp_search.web_search(
                    query, count=count, timeout=timeout
                )
        else:
            results = await freeserp_search.web_search(
                query, count=count, timeout=timeout
            )
        return _from_freeserp(results)
    except Exception:
        logger.warning("FreeSerp provider failed for query %r", query, exc_info=True)
        return []


def merge_results(
    result_lists: list[list[UnifiedSearchResult]], limit: int
) -> list[UnifiedSearchResult]:
    """Merge provider result lists, deduping by domain (first provider wins)."""
    seen: set[str] = set()
    merged: list[UnifiedSearchResult] = []
    for results in result_lists:
        for r in results:
            key = r.domain or r.url
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
            if len(merged) >= limit:
                return merged
    return merged


# --- Public API ---------------------------------------------------------------


async def web_search(
    query: str,
    *,
    count: int = DEFAULT_RESULT_COUNT,
    providers: str = "auto",
    timeout: float = 10.0,
) -> list[UnifiedSearchResult]:
    """Search the web across providers and return merged, deduped results.

    ``providers``: 'auto' (intent-routed), 'brave', 'freeserp', or 'both'.
    Never raises for provider problems: a failing provider contributes
    nothing instead.
    """
    count = max(1, min(int(count or DEFAULT_RESULT_COUNT), MAX_RESULT_COUNT))
    names = select_providers(query, override=providers if providers != "auto" else None)
    if not names:
        logger.info("No web-search provider available for query %r", query)
        return []

    intent = detect_intent(query)
    tasks = []
    for name in names:
        if name == "brave":
            tasks.append(_search_brave(query, count, timeout))
        else:
            tasks.append(
                _search_freeserp(query, count, timeout, ai_startups=intent == "discovery")
            )
    lists = await asyncio.gather(*tasks)
    return merge_results(list(lists), count)


def format_results_for_llm(query: str, results: list[UnifiedSearchResult]) -> str:
    """Format results as a system-message block for the chat model."""
    providers = sorted({r.provider for r in results if r.provider})
    via = " and ".join(p.capitalize() for p in providers) or "web search"
    lines = [
        f'Web search results for "{query}" (via {via}). '
        "Use them to answer questions about current or external facts, "
        "and cite sources inline as [1], [2], etc."
    ]
    for i, r in enumerate(results, 1):
        block = f"[{i}] {r.title} [{r.provider}]\n{r.url}"
        meta: list[str] = []
        if r.authority is not None:
            meta.append(f"DR {r.authority}")
        if r.published:
            meta.append(r.published)
        if meta:
            block += f"\n({', '.join(meta)})"
        if r.summary:
            block += f"\n{r.summary}"
        lines.append(block)
    return "\n\n".join(lines)


def should_search(messages: list[dict[str, Any]]) -> Optional[str]:
    """Return the query to search, or None when chat should answer directly.

    Conservative on purpose: only fires on an explicit search request or a
    strong recency signal, so ordinary chit-chat never pays search latency.
    """
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
    "UnifiedSearchResult",
    "detect_intent",
    "format_results_for_llm",
    "merge_results",
    "provider_override",
    "select_providers",
    "should_search",
    "web_search",
    "web_search_enabled",
]
