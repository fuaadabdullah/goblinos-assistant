"""Unit tests for backend.services.brave_search.

Network-free: parsing/heuristic tests use fixtures, and web_search tests use
a fake async client instead of httpx.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.services import brave_search
from backend.services.brave_search import (
    BraveSearchNotConfigured,
    _parse_results,
    format_results_for_llm,
    should_search,
    web_search,
)


BRAVE_FIXTURE = {
    "web": {
        "results": [
            {
                "title": "Example Result",
                "url": "https://example.com/article",
                "description": "A short snippet about the topic.",
                "page_age": "2026-09-20",
            },
            {
                "title": "",
                "url": "https://example.com/untitled",
                "description": "",
            },
            {"title": "No URL here", "description": "skipped"},
            "not-a-dict",
        ]
    }
}


def test_parse_results_normalizes_brave_payload():
    results = _parse_results(BRAVE_FIXTURE, 10)
    assert len(results) == 2  # entries without a URL are skipped
    first = results[0]
    assert first.title == "Example Result"
    assert first.url == "https://example.com/article"
    assert first.snippet == "A short snippet about the topic."
    assert first.published == "2026-09-20"
    # Empty title falls back to the URL
    assert results[1].title == "https://example.com/untitled"


def test_parse_results_respects_limit_and_bad_input():
    assert _parse_results(BRAVE_FIXTURE, 1)[0].title == "Example Result"
    assert _parse_results({}, 5) == []
    assert _parse_results(None, 5) == []
    assert _parse_results({"web": None}, 5) == []


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


def test_should_search_explicit_request():
    old = _with_env(BRAVE_API_KEY="test-key", CHAT_WEB_SEARCH_ENABLED="true")
    try:
        assert should_search([{"role": "user", "content": "Search the web for 350Z prices"}])
    finally:
        _restore_env(old)


def test_should_search_recency_signal():
    old = _with_env(BRAVE_API_KEY="test-key", CHAT_WEB_SEARCH_ENABLED="true")
    try:
        assert should_search([{"role": "user", "content": "What is the latest iPhone news?"}])
        assert should_search([{"role": "user", "content": "Who won the game last night?"}])
    finally:
        _restore_env(old)


def test_should_search_ignores_chitchat_and_missing_key():
    old = _with_env(BRAVE_API_KEY="test-key", CHAT_WEB_SEARCH_ENABLED="true")
    try:
        assert should_search([{"role": "user", "content": "Hey, how are you?"}]) is None
        assert should_search([]) is None
    finally:
        _restore_env(old)
    old = _with_env(BRAVE_API_KEY=None, CHAT_WEB_SEARCH_ENABLED="true")
    try:
        assert should_search([{"role": "user", "content": "Search the web for cats"}]) is None
    finally:
        _restore_env(old)


def test_should_search_respects_kill_switch():
    old = _with_env(BRAVE_API_KEY="test-key", CHAT_WEB_SEARCH_ENABLED="false")
    try:
        assert should_search([{"role": "user", "content": "Search the web for cats"}]) is None
    finally:
        _restore_env(old)


def test_format_results_for_llm_cites_sources():
    results = _parse_results(BRAVE_FIXTURE, 10)
    text = format_results_for_llm("test query", results)
    assert "[1]" in text and "[2]" in text
    assert "https://example.com/article" in text


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

    async def get(self, url, params=None, headers=None):
        self.seen = {"url": url, "params": params, "headers": headers}
        return _FakeResponse(self._payload)


def test_web_search_sends_subscription_token_header():
    client = _FakeClient(BRAVE_FIXTURE)
    results = asyncio.run(web_search("350Z", api_key="secret-key", client=client))
    assert client.seen["url"] == brave_search.BRAVE_SEARCH_ENDPOINT
    assert client.seen["headers"]["X-Subscription-Token"] == "secret-key"
    assert client.seen["params"]["q"] == "350Z"
    assert len(results) == 2


def test_web_search_requires_key():
    old = _with_env(BRAVE_API_KEY=None)
    try:
        with pytest.raises(BraveSearchNotConfigured):
            asyncio.run(web_search("x", client=_FakeClient({})))
    finally:
        _restore_env(old)


def test_web_search_degrades_to_empty_on_transport_error():
    class _BoomClient:
        async def get(self, *a, **k):
            raise RuntimeError("boom")

    results = asyncio.run(web_search("x", api_key="k", client=_BoomClient()))
    assert results == []
