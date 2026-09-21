"""Unit tests for backend.services.web_search.

Network-free: provider coroutines are monkeypatched with fakes; intent
detection, selection, merging, and formatting are pure functions.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.services import web_search
from backend.services.web_search import (
    UnifiedSearchResult,
    detect_intent,
    format_results_for_llm,
    merge_results,
    select_providers,
    should_search,
    web_search as facade_search,
)
from backend.services.brave_search import BraveSearchResult
from backend.services.freeserp_search import FreeSerpResult


def _with_env(**kwargs):
    old = {k: os.environ.get(k) for k in kwargs}
    os.environ.update({k: v for k, v in kwargs.items() if v is not None})
    for k, v in kwargs.items():
        if v is None:
            os.environ.pop(k, None)
    return old


def _restore_env(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# --- intent detection ---------------------------------------------------------


def test_detect_intent_discovery():
    assert detect_intent("ai voice agent startups") == "discovery"
    assert detect_intent("competitors of Notion") == "discovery"
    assert detect_intent("best alternatives to Slack") == "discovery"
    assert detect_intent("new AI companies in legal tech") == "discovery"


def test_detect_intent_recency():
    assert detect_intent("latest iPhone news") == "recency"
    assert detect_intent("What happened today in markets?") == "recency"


def test_detect_intent_general():
    assert detect_intent("how to bake sourdough bread") == "general"
    assert detect_intent("python list comprehension syntax") == "general"


# --- provider selection -------------------------------------------------------


def test_select_providers_discovery_uses_freeserp():
    old = _with_env(BRAVE_API_KEY="k", CHAT_WEB_SEARCH_PROVIDER="auto")
    try:
        assert select_providers("ai voice agent startups") == ["freeserp"]
    finally:
        _restore_env(old)


def test_select_providers_recency_uses_brave_when_keyed():
    old = _with_env(BRAVE_API_KEY="k", CHAT_WEB_SEARCH_PROVIDER="auto")
    try:
        assert select_providers("latest iPhone news") == ["brave"]
    finally:
        _restore_env(old)


def test_select_providers_recency_skips_brave_without_key():
    old = _with_env(BRAVE_API_KEY=None, CHAT_WEB_SEARCH_PROVIDER="auto")
    try:
        # FreeSerp alone is not the intent match, so nothing is selected;
        # the caller degrades gracefully to no search rather than a
        # mismatched provider.
        assert select_providers("latest iPhone news") == []
    finally:
        _restore_env(old)


def test_select_providers_general_fans_out_when_keyed():
    old = _with_env(BRAVE_API_KEY="k", CHAT_WEB_SEARCH_PROVIDER="auto")
    try:
        assert select_providers("how to bake bread") == ["brave", "freeserp"]
    finally:
        _restore_env(old)


def test_select_providers_general_keyless_is_freeserp_only():
    old = _with_env(BRAVE_API_KEY=None, CHAT_WEB_SEARCH_PROVIDER="auto")
    try:
        assert select_providers("how to bake bread") == ["freeserp"]
    finally:
        _restore_env(old)


def test_select_providers_explicit_override():
    old = _with_env(BRAVE_API_KEY=None, CHAT_WEB_SEARCH_PROVIDER="auto")
    try:
        assert select_providers("anything", override="both") == ["freeserp"]
        assert select_providers("anything", override="freeserp") == ["freeserp"]
        # Explicit brave override without a key selects nothing; the facade
        # degrades gracefully instead of raising.
        assert select_providers("anything", override="brave") == []
    finally:
        _restore_env(old)


# --- merging ------------------------------------------------------------------


def _u(title, url, provider, domain=""):
    return UnifiedSearchResult(title=title, url=url, provider=provider, domain=domain)


def test_merge_results_dedupes_by_domain_first_wins():
    brave = [_u("B1", "https://example.com/a", "brave", "example.com")]
    free = [
        _u("F1", "https://example.com/b", "freeserp", "example.com"),
        _u("F2", "https://other.io", "freeserp", "other.io"),
    ]
    merged = merge_results([brave, free], 10)
    assert [r.title for r in merged] == ["B1", "F2"]


def test_merge_results_respects_limit():
    lists = [[_u(f"t{i}", f"https://d{i}.com", "brave", f"d{i}.com") for i in range(5)]]
    assert len(merge_results(lists, 3)) == 3


# --- end-to-end with faked providers ------------------------------------------

_BRAVE_FAKE = [
    BraveSearchResult(title="Brave One", url="https://brave1.com", snippet="snip 1"),
    BraveSearchResult(title="Shared", url="https://shared.com/x", snippet="brave copy"),
]
_FREESERP_FAKE = [
    FreeSerpResult(
        domain="shared.com",
        url="https://shared.com/y",
        title="Shared",
        ai_summary="freeserp summary",
        dr=20,
        went_live="2026-08-01",
    ),
    FreeSerpResult(
        domain="free2.ai",
        url="https://free2.ai",
        title="Free Two",
        ai_summary="another summary",
        dr=35,
    ),
]


def _patch_providers(monkeypatch, brave=None, freeserp=None, startups=None):
    async def fake_brave(query, **kw):
        if isinstance(brave, Exception):
            raise brave
        return brave if brave is not None else []

    async def fake_freeserp(query, **kw):
        if isinstance(freeserp, Exception):
            raise freeserp
        return freeserp if freeserp is not None else []

    async def fake_startups(**kw):
        if isinstance(startups, Exception):
            raise startups
        return startups if startups is not None else []

    monkeypatch.setattr(web_search.brave_search, "web_search", fake_brave)
    monkeypatch.setattr(web_search.freeserp_search, "web_search", fake_freeserp)
    monkeypatch.setattr(web_search.freeserp_search, "search_ai_startups", fake_startups)


def test_facade_both_mode_merges_and_dedupes(monkeypatch):
    _patch_providers(monkeypatch, brave=_BRAVE_FAKE, freeserp=_FREESERP_FAKE)
    old = _with_env(BRAVE_API_KEY="k")
    try:
        results = asyncio.run(
            facade_search("general question", count=5, providers="both")
        )
    finally:
        _restore_env(old)
    # shared.com appears once (brave copy wins), total 3 unique domains.
    assert [r.title for r in results] == ["Brave One", "Shared", "Free Two"]
    assert results[0].provider == "brave"
    assert results[2].provider == "freeserp"
    assert results[2].authority == 35


def test_facade_survives_one_provider_failing(monkeypatch):
    _patch_providers(
        monkeypatch, brave=RuntimeError("boom"), freeserp=_FREESERP_FAKE
    )
    old = _with_env(BRAVE_API_KEY="k")
    try:
        results = asyncio.run(
            facade_search("general question", count=5, providers="both")
        )
    finally:
        _restore_env(old)
    assert len(results) == 2
    assert all(r.provider == "freeserp" for r in results)


def test_facade_no_providers_available_returns_empty(monkeypatch):
    _patch_providers(monkeypatch)
    old = _with_env(BRAVE_API_KEY=None, CHAT_WEB_SEARCH_PROVIDER="auto")
    try:
        # recency intent wants brave only; no key -> no providers -> []
        results = asyncio.run(facade_search("latest iPhone news"))
    finally:
        _restore_env(old)
    assert results == []


def test_facade_discovery_uses_ai_startups(monkeypatch):
    seen = {}

    async def fake_startups(**kw):
        seen.update(kw)
        return _FREESERP_FAKE

    async def fake_brave(query, **kw):  # pragma: no cover - must not be called
        raise AssertionError("brave should not be called for discovery")

    monkeypatch.setattr(web_search.freeserp_search, "search_ai_startups", fake_startups)
    monkeypatch.setattr(web_search.brave_search, "web_search", fake_brave)
    old = _with_env(BRAVE_API_KEY="k", CHAT_WEB_SEARCH_PROVIDER="auto")
    try:
        results = asyncio.run(facade_search("ai voice agent startups", count=4))
    finally:
        _restore_env(old)
    assert seen.get("count") == 4
    assert len(results) == 2
    assert all(r.provider == "freeserp" for r in results)


# --- trigger + formatting -----------------------------------------------------


def test_should_search_respects_kill_switch_and_chitchat():
    old = _with_env(CHAT_WEB_SEARCH_ENABLED="true")
    try:
        assert should_search([{"role": "user", "content": "search the web for cats"}])
        assert should_search([{"role": "user", "content": "hello there"}]) is None
    finally:
        _restore_env(old)
    old = _with_env(CHAT_WEB_SEARCH_ENABLED="false")
    try:
        assert should_search([{"role": "user", "content": "search the web for cats"}]) is None
    finally:
        _restore_env(old)


def test_format_results_for_llm_tags_providers():
    results = [
        UnifiedSearchResult(
            title="Brave One", url="https://brave1.com", summary="s1",
            provider="brave",
        ),
        UnifiedSearchResult(
            title="Free Two", url="https://free2.ai", summary="s2",
            provider="freeserp", authority=35, published="2026-08-01",
        ),
    ]
    text = format_results_for_llm("q", results)
    assert "[1] Brave One [brave]" in text
    assert "[2] Free Two [freeserp]" in text
    assert "via Brave and Freeserp" in text
    assert "DR 35" in text
