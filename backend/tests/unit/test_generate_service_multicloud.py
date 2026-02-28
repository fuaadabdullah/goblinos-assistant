from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest
from fastapi import HTTPException

from backend.services.generate_models import GenerateRequest
from backend.services.generate_service import generate_completion, reset_provider_state
import backend.services.generate_service as generate_service
import backend.services.generate_providers as generate_providers


class _FakeResponse:
    def __init__(self, url: str, status_code: int, payload: Dict[str, Any]):
        self._url = url
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        request = httpx.Request("POST", self._url)
        response = httpx.Response(
            self.status_code,
            request=request,
            json=self._payload,
        )
        raise httpx.HTTPStatusError(
            f"HTTP {self.status_code}",
            request=request,
            response=response,
        )


@pytest.fixture(autouse=True)
def _reset_provider_state_fixture():
    reset_provider_state()
    yield
    reset_provider_state()


@pytest.fixture(autouse=True)
def _disable_warmup(monkeypatch):
    async def _noop_warm() -> None:
        return None

    monkeypatch.setattr(generate_service, "_maybe_warm_gcp_providers", _noop_warm)


@pytest.mark.asyncio
async def test_provider_alias_normalization_routes_to_azure(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")
    monkeypatch.setenv("AZURE_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_DEPLOYMENT_ID", "chat-prod")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-08-01-preview")

    captured: dict[str, Any] = {}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers or {}
            return _FakeResponse(
                url=url,
                status_code=200,
                payload={
                    "id": "chatcmpl-1",
                    "choices": [
                        {
                            "message": {"content": "hello from azure"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = await generate_completion(
        request=GenerateRequest(
            messages=[{"role": "user", "content": "Hi"}],
            provider="azure-openai",
            model="gpt-4o-mini",
        )
    )

    assert result["provider"] == "azure_openai"
    assert result["content"] == "hello from azure"
    assert (
        captured["url"]
        == "https://azure.example/openai/deployments/chat-prod/chat/completions?api-version=2024-08-01-preview"
    )
    assert captured["headers"]["api-key"] == "azure-key"


@pytest.mark.asyncio
async def test_aliyun_openai_and_ollama_fallback_paths(monkeypatch):
    monkeypatch.setenv("ALIYUN_MODEL_SERVER_URL", "https://aliyun.example")
    monkeypatch.setenv("ALIYUN_MODEL_SERVER_KEY", "aliyun-key")

    calls: list[str] = []

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers=None, json=None):
            calls.append(url)
            if url.endswith("/v1/chat/completions"):
                return _FakeResponse(url=url, status_code=404, payload={})
            if url.endswith("/chat/completions"):
                return _FakeResponse(url=url, status_code=404, payload={})
            if url.endswith("/api/chat"):
                return _FakeResponse(url=url, status_code=404, payload={})
            if url.endswith("/api/generate"):
                return _FakeResponse(
                    url=url,
                    status_code=200,
                    payload={
                        "model": "qwen2.5:3b",
                        "response": "hello from aliyun",
                        "prompt_eval_count": 5,
                        "eval_count": 7,
                    },
                )
            raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = await generate_completion(
        request=GenerateRequest(
            messages=[{"role": "user", "content": "Hi"}],
            provider="alibaba",
            model="qwen2.5:3b",
        )
    )

    assert result["provider"] == "aliyun"
    assert result["content"] == "hello from aliyun"
    assert calls[-1].endswith("/api/generate")


@pytest.mark.asyncio
async def test_gemini_429_immediate_failover_and_cooldown(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_GCP_URL", raising=False)
    monkeypatch.delenv("LLAMACPP_GCP_URL", raising=False)
    monkeypatch.delenv("ALIYUN_MODEL_SERVER_URL", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_DEPLOYMENT_ID", raising=False)

    calls = {"gemini": 0, "openai": 0}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers=None, json=None):
            if "generativelanguage.googleapis.com" in url:
                calls["gemini"] += 1
                return _FakeResponse(url=url, status_code=429, payload={"error": "rate"})
            if "api.openai.com" in url:
                calls["openai"] += 1
                return _FakeResponse(
                    url=url,
                    status_code=200,
                    payload={
                        "model": "gpt-4o-mini",
                        "choices": [{"message": {"content": "openai fallback"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )
            raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    first = await generate_completion(
        request=GenerateRequest(messages=[{"role": "user", "content": "Hi"}])
    )
    second = await generate_completion(
        request=GenerateRequest(messages=[{"role": "user", "content": "Hi again"}])
    )

    assert first["provider"] == "openai"
    assert second["provider"] == "openai"
    assert calls["gemini"] == 1
    assert calls["openai"] == 2


@pytest.mark.asyncio
async def test_ollama_uses_simple_profile_timeout_budget(monkeypatch):
    monkeypatch.setenv("OLLAMA_GCP_URL", "https://ollama-gcp.example")
    monkeypatch.setattr(generate_service, "_GENERATE_OUTAGE_FALLBACK", False)
    timeout_capture: dict[str, int] = {}

    class _FailingAdapter:
        def __init__(self, *args, **kwargs):
            self.timeout = 0

        async def generate(self, *args, **kwargs):
            timeout_capture["timeout"] = self.timeout
            raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(generate_providers, "OllamaAdapter", _FailingAdapter)

    with pytest.raises(HTTPException):
        await generate_completion(
            request=GenerateRequest(
                messages=[{"role": "user", "content": "Hi"}], provider="ollama-gcp"
            )
        )

    # Explicit provider selection gets the forced-provider timeout budget.
    assert timeout_capture["timeout"] == 15


@pytest.mark.asyncio
async def test_openrouter_empty_content_is_treated_as_failure(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setattr(generate_service, "_GENERATE_OUTAGE_FALLBACK", False)

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers=None, json=None):
            if "openrouter.ai" not in url:
                raise AssertionError(f"unexpected URL {url}")
            return _FakeResponse(
                url=url,
                status_code=200,
                payload={
                    "model": "openrouter/auto",
                    "choices": [
                        {
                            "message": {"content": "   "},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(HTTPException):
        await generate_completion(
            request=GenerateRequest(
                messages=[{"role": "user", "content": "Hi"}], provider="openrouter"
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "env_key", "url_fragment"),
    [
        ("openai", "OPENAI_API_KEY", "api.openai.com"),
        ("deepseek", "DEEPSEEK_API_KEY", "api.deepseek.com"),
        ("anthropic", "ANTHROPIC_API_KEY", "api.anthropic.com"),
    ],
)
async def test_auth_401_triggers_cooldown_for_core_cloud_providers(
    monkeypatch, provider: str, env_key: str, url_fragment: str
):
    monkeypatch.setenv(env_key, "bad-key")

    calls = {"count": 0}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, headers=None, json=None):
            if url_fragment not in url:
                raise AssertionError(f"unexpected URL {url}")
            calls["count"] += 1
            return _FakeResponse(url=url, status_code=401, payload={"error": "unauthorized"})

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(HTTPException):
        await generate_completion(
            request=GenerateRequest(
                messages=[{"role": "user", "content": "Hi"}], provider=provider
            )
        )
    with pytest.raises(HTTPException):
        await generate_completion(
            request=GenerateRequest(
                messages=[{"role": "user", "content": "Hi again"}], provider=provider
            )
        )

    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_outage_fallback_response_when_enabled(monkeypatch):
    monkeypatch.setattr(generate_service, "_GENERATE_OUTAGE_FALLBACK", True)
    for key in [
        "OLLAMA_GCP_URL",
        "OLLAMA_URL",
        "OLLAMA_BASE_URL",
        "LLAMACPP_GCP_URL",
        "LLAMACPP_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "SILICONEFLOW_API_KEY",
        "SILLICONFLOW_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_API_KEY",
        "AZURE_DEPLOYMENT_ID",
        "ALIYUN_MODEL_SERVER_URL",
    ]:
        monkeypatch.delenv(key, raising=False)

    result = await generate_completion(
        request=GenerateRequest(messages=[{"role": "user", "content": "Hi"}])
    )

    assert result["provider"] == "fallback_unavailable"
    assert "temporarily unavailable" in str(result["content"]).lower()
