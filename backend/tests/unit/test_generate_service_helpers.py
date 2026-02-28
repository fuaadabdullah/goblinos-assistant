from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import HTTPException

from backend.services import generate_service as gs
from backend.services.generate_models import GenerateRequest
from backend.services.generate_providers import load_provider_config


def _reset_provider_state() -> None:
    gs._provider_health_cache.clear()
    gs._provider_failure_counts.clear()
    gs._provider_circuit_open_until.clear()
    gs._provider_auth_blocked_until.clear()
    gs._provider_rate_limited_until.clear()


def test_is_simple_prompt_variants() -> None:
    assert gs._is_simple_prompt([{"role": "user", "content": "hi"}]) is True
    assert (
        gs._is_simple_prompt(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "short"},
            ]
        )
        is True
    )
    assert gs._is_simple_prompt([{"role": "user", "content": ""}]) is False
    assert gs._is_simple_prompt([{"role": "user", "content": "x" * 33}]) is False
    assert (
        gs._is_simple_prompt(
            [
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "again"},
            ]
        )
        is False
    )


def test_provider_cache_success_resets(monkeypatch) -> None:
    _reset_provider_state()
    monkeypatch.setattr(gs.time, "time", lambda: 1000.0)
    gs._provider_failure_counts["p"] = 2
    gs._provider_circuit_open_until["p"] = 9999.0
    gs._provider_auth_blocked_until["p"] = 9999.0
    gs._provider_rate_limited_until["p"] = 9999.0

    gs._mark_provider_success("p")

    assert gs._provider_health_cache["p"] == (1000.0, True)
    assert gs._provider_failure_counts["p"] == 0
    assert "p" not in gs._provider_circuit_open_until
    assert "p" not in gs._provider_auth_blocked_until
    assert "p" not in gs._provider_rate_limited_until


def test_provider_failure_opens_circuit(monkeypatch) -> None:
    _reset_provider_state()
    monkeypatch.setattr(gs.time, "time", lambda: 2000.0)

    gs._mark_provider_failure("p", retryable=True)
    gs._mark_provider_failure("p", retryable=True)
    gs._mark_provider_failure("p", retryable=True)

    assert gs._provider_failure_counts["p"] >= gs._PROVIDER_CIRCUIT_FAILS
    assert gs._provider_circuit_open_until["p"] > 2000.0
    assert gs._is_provider_blocked("p") is True

    monkeypatch.setattr(
        gs.time,
        "time",
        lambda: 2000.0 + gs._PROVIDER_CIRCUIT_COOLDOWN_S + 1.0,
    )
    assert gs._is_provider_blocked("p") is False


def test_provider_recently_unhealthy_ttl(monkeypatch) -> None:
    _reset_provider_state()
    monkeypatch.setattr(gs.time, "time", lambda: 3000.0)
    gs._provider_health_cache["p"] = (3000.0, False)

    assert gs._provider_recently_unhealthy("p") is True

    monkeypatch.setattr(gs.time, "time", lambda: 3000.0 + gs._PROVIDER_HEALTH_TTL_S + 1.0)
    assert gs._provider_recently_unhealthy("p") is False


def test_provider_auth_block(monkeypatch) -> None:
    _reset_provider_state()
    monkeypatch.setattr(gs.time, "time", lambda: 4000.0)
    gs._mark_provider_auth_failure("p")

    assert gs._is_provider_auth_blocked("p") is True

    monkeypatch.setattr(
        gs.time,
        "time",
        lambda: 4000.0 + gs._PROVIDER_AUTH_COOLDOWN_S + 1.0,
    )
    assert gs._is_provider_auth_blocked("p") is False


def test_provider_rate_limit_block(monkeypatch) -> None:
    _reset_provider_state()
    monkeypatch.setattr(gs.time, "time", lambda: 4500.0)
    gs._mark_provider_rate_limited("p")

    assert gs._is_provider_rate_limited("p") is True

    monkeypatch.setattr(
        gs.time,
        "time",
        lambda: 4500.0 + gs._PROVIDER_RATE_LIMIT_COOLDOWN_S + 1.0,
    )
    assert gs._is_provider_rate_limited("p") is False


def test_provider_alias_normalization() -> None:
    assert gs._normalize_provider_id("ollama-gcp") == "ollama_gcp"
    assert gs._normalize_provider_id("azure-openai") == "azure_openai"
    assert gs._normalize_provider_id("alibaba") == "aliyun"
    assert gs._normalize_provider_id("openai_fallback") == "openai"


def test_provider_unknown_id_passes_through_when_not_recognized() -> None:
    value = gs._normalize_provider_id(
        "my-custom-provider",
        known_provider_ids={"openai", "azure_openai"},
    )
    assert value == "my-custom-provider"


def test_derive_generation_params_uses_dynamic_defaults_and_2048_clamp() -> None:
    class _Request:
        def __init__(self, max_tokens=None, temperature=None):
            self.max_tokens = max_tokens
            self.temperature = temperature

    short_tokens, short_temp = gs._derive_generation_params(_Request(), "hi")
    medium_tokens, medium_temp = gs._derive_generation_params(
        _Request(),
        "x" * 100,
    )
    long_tokens, long_temp = gs._derive_generation_params(
        _Request(),
        "x" * 400,
    )
    clamped_tokens, clamped_temp = gs._derive_generation_params(
        _Request(max_tokens=9999, temperature=9.9),
        "anything",
    )

    assert short_tokens == 256
    assert medium_tokens == 768
    assert long_tokens == 1536
    assert short_temp == 0.2
    assert medium_temp == 0.2
    assert long_temp == 0.2
    assert clamped_tokens == 2048
    assert clamped_temp == 2.0


def test_retryable_error_detection() -> None:
    assert gs._is_retryable_error(httpx.ConnectTimeout("timeout")) is True
    assert gs._is_retryable_error(httpx.ReadTimeout("timeout")) is True
    assert gs._is_retryable_error(httpx.WriteTimeout("timeout")) is True
    assert gs._is_retryable_error(httpx.ConnectError("connect")) is True
    assert gs._is_retryable_error(RuntimeError("boom")) is False


def test_provider_config_supports_cloud_env_aliases(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_KEY", "router-key")
    monkeypatch.setenv("OLLAMA_GCP_BASE_URL", "http://gcp-ollama:11434")
    monkeypatch.delenv("OLLAMA_GCP_URL", raising=False)
    monkeypatch.setenv("LOCAL_LLM_PROXY_URL", "http://proxy:11434")
    monkeypatch.setenv("GCP_LLAMACPP_URL", "http://gcp-llamacpp:8080")
    monkeypatch.delenv("LLAMACPP_GCP_URL", raising=False)
    monkeypatch.delenv("LLAMACPP_GCP_BASE_URL", raising=False)
    monkeypatch.setenv("LLAMACPP_URL", "http://llamacpp-local:8080")
    monkeypatch.delenv("SILICONEFLOW_API_KEY", raising=False)
    monkeypatch.setenv("SILLICONFLOW_API_KEY", "silicone-legacy")

    config = load_provider_config()

    assert config.openrouter_key == "router-key"
    assert config.ollama_url == "http://gcp-ollama:11434"
    assert config.llamacpp_url == "http://gcp-llamacpp:8080"
    assert config.siliconeflow_key == "silicone-legacy"


@pytest.mark.asyncio
async def test_maybe_warm_gcp_providers_once(monkeypatch) -> None:
    _reset_provider_state()
    calls: list[str] = []

    async def _fake_warm() -> None:
        calls.append("warm")

    monkeypatch.setattr(gs, "_warm_gcp_providers_once", _fake_warm)
    monkeypatch.setattr(gs.time, "time", lambda: 5000.0)
    gs._gcp_warm_last_at = 0.0

    await gs._maybe_warm_gcp_providers()
    await gs._maybe_warm_gcp_providers()

    assert calls == ["warm"]


def test_provider_strategy_excludes_non_cloud_local_provider() -> None:
    simple_order, _ = gs._build_provider_strategy(simple_prompt=True, forced_provider=None)
    complex_order, _ = gs._build_provider_strategy(
        simple_prompt=False, forced_provider=None
    )

    assert "goblin-chat" not in simple_order
    assert "goblin-chat" not in complex_order
    assert simple_order[:4] == ["aliyun", "azure_openai", "ollama_gcp", "llamacpp_gcp"]
    assert complex_order[0] == "aliyun"


def test_simple_prompt_strategy_uses_low_latency_timeouts() -> None:
    _order, profiles = gs._build_provider_strategy(simple_prompt=True, forced_provider=None)

    assert profiles["ollama_gcp"] == (1.6, 0.6)
    assert profiles["llamacpp_gcp"] == (1.6, 0.6)
    assert profiles["aliyun"] == (1.6, 0.6)
    assert profiles["azure_openai"] == (1.2, 0.6)
    assert profiles["openai"] == (1.1, 0.6)


def test_provider_config_remote_gcp_remains_configured_when_local_ml_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DISABLE_LOCAL_ML", "true")
    monkeypatch.setenv("OLLAMA_GCP_URL", "http://ollama-gcp.example")
    monkeypatch.setenv("LLAMACPP_GCP_URL", "http://llamacpp-gcp.example")

    config = load_provider_config()

    assert gs._provider_is_configured("ollama_gcp", config) is True
    assert gs._provider_is_configured("llamacpp_gcp", config) is True


def test_provider_config_azure_services_endpoint_does_not_require_deployment() -> None:
    class _Config:
        azure_openai_endpoint = "https://goblinos-resource.services.ai.azure.com"
        azure_api_key = "azure-key"
        azure_deployment_id = ""

    assert gs._provider_is_configured("azure_openai", _Config()) is True


@pytest.mark.asyncio
async def test_attempt_provider_respects_request_deadline() -> None:
    _reset_provider_state()
    errors: list[str] = []

    async def _slow_call():
        await asyncio.sleep(1.0)
        return {"content": "late"}

    result = await gs._attempt_provider(
        provider_name="openrouter",
        provider_call=_slow_call,
        max_attempts=1,
        request_deadline=gs.time.time() + 0.01,
        errors=errors,
        provider_labels={"openrouter": "OpenRouter"},
        correlation_id="cid-timeout",
    )

    assert result is None
    assert errors == ["OpenRouter: timeout"]


@pytest.mark.asyncio
async def test_explicit_provider_selection_returns_503_when_provider_fails(monkeypatch) -> None:
    _reset_provider_state()
    monkeypatch.setattr(gs, "_GENERATE_OUTAGE_FALLBACK", True)

    async def _noop_warm() -> None:
        return None

    class _FakeRegistry:
        def __init__(self):
            self.providers = {"openai": object()}

        def get_provider_catalog(self):
            return {"openai": {"models": ["gpt-4o-mini"]}}

        def get_provider_health_map(self, refresh=True):
            return {}

    class _FakeConfig:
        openai_key = "test-key"
        siliconeflow_key = None
        ollama_url = None
        llamacpp_url = None
        gemini_key = None
        deepseek_key = None
        openrouter_key = None
        anthropic_key = None
        azure_openai_endpoint = None
        azure_api_key = None
        azure_deployment_id = None
        aliyun_url = None
        groq_key = None

    async def _failing_openai():
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(gs, "_maybe_warm_gcp_providers", _noop_warm)
    monkeypatch.setattr(gs, "get_provider_registry", lambda: _FakeRegistry())
    monkeypatch.setattr(gs, "load_provider_config", lambda: _FakeConfig())
    monkeypatch.setattr(gs, "build_provider_attempts", lambda *_args, **_kwargs: {"openai": _failing_openai})

    with pytest.raises(HTTPException) as exc_info:
        await gs.generate_completion(
            request=GenerateRequest(
                messages=[{"role": "user", "content": "Hi"}],
                provider="openai",
                model="gpt-4o-mini",
            )
        )

    assert exc_info.value.status_code == 503
