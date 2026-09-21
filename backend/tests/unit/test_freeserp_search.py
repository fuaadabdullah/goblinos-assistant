"""Unit tests for backend.services.freeserp_search.

Network-free: parsing/heuristic tests use fixtures, and web_search tests use
a fake async client instead of httpx.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.services import freeserp_search
from backend.services.freeserp_search import (
    _parse_results,
    format_results_for_llm,
    provider_name,
    search_ai_startups,
    should_search,
    web_search,
)


FREESERP_FIXTURE = {
    "ok": True,
    "query": "ai chatbot",
    "total": 22135,
    "count": 3,
    "filters": {"real_site": "1"},
    "results": [
        {
            "domain": "example.ai",
            "url": "https://example.ai",
            "title": "Example — AI chatbot for support",
            "ai_summary": "Example is an AI chatbot platform that answers support tickets.",
            "category": "ai",
            "ai_categories": ["AI Chatbot & Assistant"],
            "dr": 41,
            "went_live": "2026-08-22",
            "real_site": 1,
        },
        {
            "domain": "untitled.example",
            "url": "https://untitled.example",
            "title": "",
            "ai_summary": "",
            "dr": None,
        },
        {"domain": "nouser.example", "title": "no url", "ai_summary": "skipped"},
        "not-a-dict",
    ],
}


def test_parse_results_normalizes_freeserp_payload():
    results = _parse_results(FREESERP_FIXTURE, 10)
    assert len(results) == 2  # entries without a URL are skipped
    first = results[0]
    assert first.domain == "example.ai"
    assert first.url == "https://example.ai"
    assert first.title == "Example — AI chatbot for support"
    assert first.ai_summary.startswith("Example is an AI chatbot")
    assert first.dr == 41
    assert first.went_live == "2026-08-22"
    assert first.ai_categories == ["AI Chatbot & Assistant"]
    # Empty title falls back to the URL
    assert results[1].title == "https://untitled.example"


def test_parse_results_respects_limit_and_bad_input():
    assert _parse_results(FREESERP_FIXTURE, 1)[0].title == "Example — AI chatbot for support"
    assert _parse_results({}, 5) == []
    assert _parse_results(None, 5) == []
    assert _parse_results({"ok": False, "results": []}, 5) == []
    assert _parse_results({"ok": True, "results": None}, 5) == []


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


def test_should_search_needs_no_api_key():
    # FreeSerp is keyless: it fires even when BRAVE_API_KEY is unset.
    old = _with_env(BRAVE_API_KEY=None, CHAT_WEB_SEARCH_ENABLED="true")
    try:
        assert should_search([{"role": "user", "content": "Search the web for 350Z prices"}])
        assert should_search([{"role": "user", "content": "What is the latest iPhone news?"}])
    finally:
        _restore_env(old)


def test_should_search_ignores_chitchat():
    old = _with_env(CHAT_WEB_SEARCH_ENABLED="true")
    try:
        assert should_search([{"role": "user", "content": "Hey, how are you?"}]) is None
        assert should_search([]) is None
    finally:
        _restore_env(old)


def test_should_search_respects_kill_switch():
    old = _with_env(CHAT_WEB_SEARCH_ENABLED="false")
    try:
        assert should_search([{"role": "user", "content": "Search the web for cats"}]) is None
    finally:
        _restore_env(old)


def test_format_results_for_llm_cites_sources():
    results = _parse_results(FREESERP_FIXTURE, 10)
    text = format_results_for_llm("test query", results)
    assert "[1]" in text and "[2]" in text
    assert "https://example.ai" in text
    assert "FreeSerp" in text
    assert "DR 41" in text
    assert "live since 2026-08-22" in text


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.seen = {}

    async def get(self, url, params=None):
        self.seen = {"url": url, "params": params}
        return _FakeResponse(self._payload)


def test_web_search_hits_endpoint_with_size_and_courtesy_agent():
    client = _FakeClient(FREESERP_FIXTURE)
    results = asyncio.run(web_search("350Z", count=7, client=client))
    assert client.seen["url"] == freeserp_search.FREESERP_ENDPOINT
    assert client.seen["params"]["q"] == "350Z"
    assert client.seen["params"]["size"] == 7
    # No key header is ever required; the courtesy identity is a param.
    assert client.seen["params"]["agent"] == "goblin-assistant"
    assert len(results) == 2


def test_web_search_ai_startups_filter():
    client = _FakeClient(FREESERP_FIXTURE)
    asyncio.run(web_search("agents", ai_startups=True, client=client))
    assert client.seen["params"]["ai_startups"] == 1


def test_web_search_degrades_to_empty_on_transport_error():
    class _BoomClient:
        async def get(self, *a, **k):
            raise RuntimeError("boom")

    results = asyncio.run(web_search("x", client=_BoomClient()))
    assert results == []


def test_search_ai_startups_browse_mode_params():
    client = _FakeClient(FREESERP_FIXTURE)
    results = asyncio.run(
        search_ai_startups(
            ai_category="AI Agents & Autonomous", from_date="2026-09-01", client=client
        )
    )
    params = client.seen["params"]
    assert params["ai_startups"] == 1
    assert params["ai_categories"] == "AI Agents & Autonomous"
    assert params["from_date"] == "2026-09-01"
    assert params["sort"] == "went_live"
    assert params["order"] == "desc"
    assert "q" not in params
    assert len(results) == 2


def test_provider_name_auto_prefers_brave_when_keyed():
    old = _with_env(BRAVE_API_KEY="test-key", CHAT_WEB_SEARCH_PROVIDER="auto")
    try:
        assert provider_name() == "brave"
    finally:
        _restore_env(old)
    old = _with_env(BRAVE_API_KEY=None, CHAT_WEB_SEARCH_PROVIDER="auto")
    try:
        assert provider_name() == "freeserp"
    finally:
        _restore_env(old)


def test_provider_name_explicit_override():
    old = _with_env(BRAVE_API_KEY="test-key", CHAT_WEB_SEARCH_PROVIDER="freeserp")
    try:
        assert provider_name() == "freeserp"
    finally:
        _restore_env(old)
