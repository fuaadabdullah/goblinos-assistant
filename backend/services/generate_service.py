import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import httpx
from fastapi import HTTPException, Response

try:  # backend package import layout
    from ..errors import ErrorCodes, raise_problem
    from ..providers.registry import get_provider_registry
    from .generate_models import GenerateRequest
    from .generate_providers import (
        ProviderContext,
        build_provider_attempts,
        load_provider_config,
    )
    from .provider_catalog import (
        build_known_provider_ids,
        build_provider_index,
        canonicalize_provider_id,
    )
    from .generate_circuit_breaker import (
        _provider_health_cache,
        _provider_failure_counts,
        _provider_circuit_open_until,
        _provider_auth_blocked_until,
        _provider_rate_limited_until,
        _PROVIDER_HEALTH_TTL_S,
        _PROVIDER_CIRCUIT_FAILS,
        _PROVIDER_CIRCUIT_COOLDOWN_S,
        _PROVIDER_AUTH_COOLDOWN_S,
        _PROVIDER_RATE_LIMIT_COOLDOWN_S,
        reset_provider_state,
        is_provider_blocked as _is_provider_blocked,
        is_provider_auth_blocked as _is_provider_auth_blocked,
        is_provider_rate_limited as _is_provider_rate_limited,
        provider_recently_unhealthy as _provider_recently_unhealthy,
        mark_provider_success as _mark_provider_success,
        mark_provider_failure as _mark_provider_failure,
        mark_provider_auth_failure as _mark_provider_auth_failure,
        mark_provider_rate_limited as _mark_provider_rate_limited,
    )
    from .generate_cost import (
        normalize_usage as _normalize_usage,
        get_usd_per_1k as _get_usd_per_1k,
        compute_cost_usd as _compute_cost_usd,
    )
    from .generate_helpers import (
        is_retryable_error as _is_retryable_error,
        is_auth_error as _is_auth_error,
        is_rate_limited_error as _is_rate_limited_error,
        safe_err as _safe_err,
        is_simple_prompt as _is_simple_prompt,
        build_outage_fallback_text as _build_outage_fallback_text,
        normalize_provider_id as _normalize_provider_id,
    )
except ImportError:  # backend root on PYTHONPATH
    from errors import ErrorCodes, raise_problem
    from providers.registry import get_provider_registry
    from services.generate_models import GenerateRequest
    from services.generate_providers import (
        ProviderContext,
        build_provider_attempts,
        load_provider_config,
    )
    from services.provider_catalog import (
        build_known_provider_ids,
        build_provider_index,
        canonicalize_provider_id,
    )
    from services.generate_circuit_breaker import (  # type: ignore[no-redef]
        _provider_health_cache,
        _provider_failure_counts,
        _provider_circuit_open_until,
        _provider_auth_blocked_until,
        _provider_rate_limited_until,
        _PROVIDER_HEALTH_TTL_S,
        _PROVIDER_CIRCUIT_FAILS,
        _PROVIDER_CIRCUIT_COOLDOWN_S,
        _PROVIDER_AUTH_COOLDOWN_S,
        _PROVIDER_RATE_LIMIT_COOLDOWN_S,
        reset_provider_state,
        is_provider_blocked as _is_provider_blocked,
        is_provider_auth_blocked as _is_provider_auth_blocked,
        is_provider_rate_limited as _is_provider_rate_limited,
        provider_recently_unhealthy as _provider_recently_unhealthy,
        mark_provider_success as _mark_provider_success,
        mark_provider_failure as _mark_provider_failure,
        mark_provider_auth_failure as _mark_provider_auth_failure,
        mark_provider_rate_limited as _mark_provider_rate_limited,
    )
    from services.generate_cost import (  # type: ignore[no-redef]
        normalize_usage as _normalize_usage,
        get_usd_per_1k as _get_usd_per_1k,
        compute_cost_usd as _compute_cost_usd,
    )
    from services.generate_helpers import (  # type: ignore[no-redef]
        is_retryable_error as _is_retryable_error,
        is_auth_error as _is_auth_error,
        is_rate_limited_error as _is_rate_limited_error,
        safe_err as _safe_err,
        is_simple_prompt as _is_simple_prompt,
        build_outage_fallback_text as _build_outage_fallback_text,
        normalize_provider_id as _normalize_provider_id,
    )

logger = logging.getLogger(__name__)

# Opportunistically "wake" self-hosted GCP providers in the background so they don't
# cold-start when they are needed. This must never block a user request.
_GCP_WARM_MIN_INTERVAL_S = float(
    os.getenv("GCP_WARMUP_INTERVAL_S") or os.getenv("GCP_WARM_MIN_INTERVAL_S") or "300"
)
_gcp_warm_last_at: float = 0.0
_gcp_warm_lock = asyncio.Lock()

_GENERATE_OUTAGE_FALLBACK = str(
    os.getenv("GENERATE_OUTAGE_FALLBACK", "false")
).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GenerateContext:
    correlation_id: str
    messages: List[Dict[str, str]]
    prompt: str
    model: str
    provider_hint: str
    req_max_tokens: int
    req_temperature: float
    simple_prompt: bool
    request_deadline: float
    forced_provider: Optional[str]
    forced_model: Optional[str]


async def _attempt_provider(
    provider_name: str,
    provider_call: Callable[[], Awaitable[Dict[str, Any]]],
    max_attempts: int,
    *,
    request_deadline: float,
    errors: List[str],
    provider_labels: Dict[str, str],
    correlation_id: str,
) -> Optional[Dict[str, Any]]:
    for attempt in range(max_attempts):
        try:
            remaining_budget = max(0.2, request_deadline - time.time())
            result = await asyncio.wait_for(provider_call(), timeout=remaining_budget)
            _mark_provider_success(provider_name)
            if isinstance(result, dict):
                # Normalize response fields for frontend visibility.
                if "content" not in result and "response" in result:
                    result["content"] = result.get("response") or ""
                result.setdefault("provider", provider_name)
                result["correlation_id"] = correlation_id

                normalized_usage = _normalize_usage(result.get("usage", {}))
                if normalized_usage:
                    result["usage"] = normalized_usage
                else:
                    result.pop("usage", None)

                # Compute cost if not already provided by upstream.
                if "cost_usd" not in result or result.get("cost_usd") is None:
                    if normalized_usage:
                        result["cost_usd"] = _compute_cost_usd(
                            normalized_usage,
                            str(result.get("provider") or provider_name),
                        )
            return result
        except asyncio.TimeoutError:
            _mark_provider_failure(provider_name, retryable=False)
            safe_error = f"{provider_labels.get(provider_name, provider_name)}: timeout"
            errors.append(safe_error)
            logger.warning(
                "Provider attempt failed: %s",
                safe_error,
                extra={
                    "correlation_id": correlation_id,
                    "provider": provider_name,
                    "error": safe_error,
                },
            )
            return None
        except Exception as e:
            auth_error = _is_auth_error(e)
            rate_limit_error = _is_rate_limited_error(e)
            if auth_error:
                _mark_provider_auth_failure(provider_name)
            if rate_limit_error:
                _mark_provider_rate_limited(provider_name)
            retryable = _is_retryable_error(e) and not auth_error and not rate_limit_error
            _mark_provider_failure(provider_name, retryable=retryable)
            safe_error = _safe_err(provider_labels.get(provider_name, provider_name), e)
            errors.append(safe_error)
            logger.warning(
                "Provider attempt failed: %s",
                safe_error,
                extra={
                    "correlation_id": correlation_id,
                    "provider": provider_name,
                    "error": safe_error,
                },
            )
            if retryable and attempt < (max_attempts - 1):
                continue
            return None
    return None


def _build_messages_and_prompt(
    request: GenerateRequest, system_prompt: str
) -> tuple[List[Dict[str, str]], str]:
    messages: List[Dict[str, str]] = [
        {"role": m.role, "content": m.content} for m in request.messages
    ]
    if not messages:
        raise HTTPException(status_code=400, detail='Missing "messages"')

    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})

    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    ).strip()
    if not last_user:
        raise HTTPException(status_code=400, detail="Missing user message content")

    return messages, last_user


def _derive_generation_params(request: GenerateRequest, last_user: str) -> tuple[int, float]:
    def _default_max_tokens(text: str) -> int:
        n = len((text or "").strip())
        if n <= 32:
            return 256
        if n <= 200:
            return 768
        return 1536

    req_max_tokens = (
        int(request.max_tokens) if request.max_tokens else _default_max_tokens(last_user)
    )
    req_max_tokens = max(1, min(req_max_tokens, 2048))

    req_temperature = (
        float(request.temperature) if request.temperature is not None else 0.2
    )
    req_temperature = max(0.0, min(req_temperature, 2.0))

    return req_max_tokens, req_temperature


def _build_provider_strategy(
    simple_prompt: bool, forced_provider: Optional[str]
) -> tuple[List[str], Dict[str, tuple[float, float]]]:
    if simple_prompt:
        provider_timeout_profiles = {
            "aliyun": (1.6, 0.6),
            "azure_openai": (1.2, 0.6),
            "ollama_gcp": (1.6, 0.6),
            "llamacpp_gcp": (1.6, 0.6),
            "gemini": (1.1, 0.6),
            "deepseek": (1.1, 0.6),
            "openrouter": (3.0, 1.0),
            "openai": (1.1, 0.6),
            "anthropic": (1.1, 0.6),
            "groq": (1.1, 0.6),
            "siliconeflow": (3.0, 1.0),
        }
        provider_order = [
            "aliyun",
            "azure_openai",
            "ollama_gcp",
            "llamacpp_gcp",
            "gemini",
            "deepseek",
            "openrouter",
            "openai",
            "anthropic",
            "groq",
            "siliconeflow",
        ]
    else:
        provider_timeout_profiles = {
            "aliyun": (6.0, 1.0),
            "azure_openai": (6.0, 1.0),
            "ollama_gcp": (8.0, 1.0),
            "llamacpp_gcp": (8.0, 1.0),
            "openai": (6.0, 1.0),
            "anthropic": (6.0, 1.0),
            "gemini": (6.0, 1.0),
            "deepseek": (6.0, 1.0),
            "openrouter": (6.0, 1.0),
            "groq": (6.0, 1.0),
            "siliconeflow": (6.0, 1.0),
        }
        provider_order = [
            "aliyun",
            "azure_openai",
            "ollama_gcp",
            "llamacpp_gcp",
            "openai",
            "anthropic",
            "gemini",
            "deepseek",
            "openrouter",
            "groq",
            "siliconeflow",
        ]

    if forced_provider:
        provider_order = [forced_provider]

    return provider_order, provider_timeout_profiles


def _remaining_timeout(
    default_total: float, request_deadline: float, default_connect: float = 2.0
) -> httpx.Timeout:
    remaining = max(0.2, request_deadline - time.time())
    return httpx.Timeout(
        min(default_total, remaining), connect=min(default_connect, remaining)
    )


def _provider_timeout(
    name: str,
    request_deadline: float,
    profiles: Dict[str, tuple[float, float]],
    forced_provider: Optional[str],
) -> httpx.Timeout:
    if forced_provider and name == forced_provider:
        return _remaining_timeout(15.0, request_deadline, default_connect=1.5)
    total, connect = profiles.get(name, (6.0, 1.0))
    return _remaining_timeout(total, request_deadline, default_connect=connect)


def _provider_is_configured(name: str, config) -> bool:
    if name == "siliconeflow":
        return bool(config.siliconeflow_key)
    if name == "ollama_gcp":
        return bool(config.ollama_url)
    if name == "llamacpp_gcp":
        return bool(config.llamacpp_url)
    if name == "gemini":
        return bool(config.gemini_key)
    if name == "deepseek":
        return bool(config.deepseek_key and config.deepseek_key != "placeholder")
    if name == "openrouter":
        return bool(config.openrouter_key and config.openrouter_key != "placeholder")
    if name == "openai":
        return bool(config.openai_key and config.openai_key != "placeholder")
    if name == "anthropic":
        return bool(config.anthropic_key and config.anthropic_key != "placeholder")
    if name == "azure_openai":
        endpoint = str(config.azure_openai_endpoint or "").strip()
        if endpoint and ".services.ai.azure.com" in endpoint.lower():
            return bool(
                endpoint
                and config.azure_api_key
                and str(config.azure_api_key).strip() != "placeholder"
            )
        return bool(
            endpoint
            and config.azure_api_key
            and config.azure_deployment_id
            and str(config.azure_api_key).strip() != "placeholder"
        )
    if name == "aliyun":
        return bool(config.aliyun_url)
    if name == "groq":
        return bool(config.groq_key and config.groq_key != "placeholder")
    return True


def _provider_allowed(
    name: str, provider_hint: str, forced_provider: Optional[str]
) -> bool:
    return _provider_block_reason(name, provider_hint, forced_provider) is None


def _provider_block_reason(
    name: str, provider_hint: str, forced_provider: Optional[str]
) -> Optional[str]:
    if provider_hint and provider_hint != name:
        return "provider_hint_mismatch"
    # Explicit selections should still respect provider health/circuit gates;
    # we fail fast with a 503 instead of silently falling back.
    if _is_provider_auth_blocked(name):
        return "auth_blocked"
    if _is_provider_rate_limited(name):
        return "rate_limited"
    if _is_provider_blocked(name):
        return "circuit_open"
    if _provider_recently_unhealthy(name):
        return "recently_unhealthy"
    return None


def _max_attempts_for_provider(
    provider_name: str, simple_prompt: bool, forced_provider: Optional[str]
) -> int:
    if (
        provider_name in {"ollama_gcp", "llamacpp_gcp"}
        and not simple_prompt
        and not forced_provider
    ):
        return 2
    return 1


async def _warm_gcp_providers_once() -> None:
    # Centralized warm-up: keepalive + optional tiny inference.
    from .gcp_warmup import warm_gcp_endpoints

    await warm_gcp_endpoints(reason="request-path")


async def _maybe_warm_gcp_providers() -> None:
    global _gcp_warm_last_at

    now = time.time()
    if (now - _gcp_warm_last_at) < _GCP_WARM_MIN_INTERVAL_S:
        return

    async with _gcp_warm_lock:
        now = time.time()
        if (now - _gcp_warm_last_at) < _GCP_WARM_MIN_INTERVAL_S:
            return
        _gcp_warm_last_at = now

    try:
        await _warm_gcp_providers_once()
    except Exception as e:
        logger.debug("GCP warm-up failed", extra={"error": type(e).__name__})


def _raise_unknown_provider_selection(
    provider: str,
    correlation_id: str,
) -> None:
    raise_problem(
        status=422,
        title="Unprocessable Entity",
        detail=f"Unknown provider selection: '{provider}'.",
        type_uri="https://goblin-backend.onrender.com/errors/invalid-provider-selection",
        code=ErrorCodes.INVALID_FIELD_VALUE,
        errors={"provider": ["Unknown provider selection."]},
        instance=correlation_id,
        headers={"X-Correlation-ID": correlation_id},
    )


def _raise_unknown_model_selection(
    provider: str,
    model: str,
    correlation_id: str,
) -> None:
    raise_problem(
        status=422,
        title="Unprocessable Entity",
        detail=f"Model '{model}' is not configured for provider '{provider}'.",
        type_uri="https://goblin-backend.onrender.com/errors/invalid-model-selection",
        code=ErrorCodes.MODEL_NOT_AVAILABLE,
        errors={"model": [f"Model '{model}' is not configured for provider '{provider}'."]},
        instance=correlation_id,
        headers={"X-Correlation-ID": correlation_id},
    )


def _raise_unavailable_provider_selection(
    provider: str,
    health: str,
    health_reason: Optional[str],
    correlation_id: str,
) -> None:
    detail = health_reason or f"Provider '{provider}' is currently {health}."
    raise_problem(
        status=503,
        title="Service Unavailable",
        detail=detail,
        type_uri="https://goblin-backend.onrender.com/errors/service-unavailable",
        code=ErrorCodes.SERVICE_UNAVAILABLE,
        instance=correlation_id,
        headers={"X-Correlation-ID": correlation_id},
    )


def _health_for_block_reason(block_reason: str) -> tuple[str, str]:
    if block_reason == "circuit_open":
        return (
            "circuit_open",
            "Provider temporarily blocked by circuit protection.",
        )
    if block_reason == "rate_limited":
        return (
            "unhealthy",
            "Provider temporarily rate-limited.",
        )
    if block_reason == "auth_blocked":
        return (
            "unhealthy",
            "Provider temporarily blocked after authentication failures.",
        )
    if block_reason == "recently_unhealthy":
        return (
            "unhealthy",
            "Provider recently failed health checks.",
        )
    return ("unhealthy", "Provider is currently unavailable.")


async def generate_completion(
    request: GenerateRequest,
    forced_provider: Optional[str] = None,
    forced_model: Optional[str] = None,
    correlation_id: Optional[str] = None,
    response: Optional[Response] = None,
):
    """Generate completion with robust provider fallback and fast-path routing."""
    correlation_id = correlation_id or str(uuid.uuid4())
    if response is not None:
        response.headers["X-Correlation-ID"] = correlation_id
    asyncio.create_task(_maybe_warm_gcp_providers())

    system_prompt = (
        os.getenv("GOBLIN_SYSTEM_PROMPT")
        or "You are Goblin Assistant. Respond as the assistant only. Do not include role labels like 'User:' or 'Assistant:'. "
        "Do not claim you performed real-world actions (sending emails/messages, payments, etc.). "
        "If asked to send a message/email, say you cannot send it directly and offer to draft it, asking for the needed details. "
        "Be concise unless the user asks for more detail."
    )

    messages, last_user = _build_messages_and_prompt(request, system_prompt)
    req_max_tokens, req_temperature = _derive_generation_params(request, last_user)

    model = (forced_model or request.model or "llama2").strip()
    registry = get_provider_registry()
    provider_index = build_provider_index(registry, refresh_health=False)
    known_provider_id_source: List[str] = list(provider_index.keys())
    get_provider_catalog = getattr(registry, "get_provider_catalog", None)
    if callable(get_provider_catalog):
        try:
            provider_catalog = get_provider_catalog() or {}
            if isinstance(provider_catalog, dict):
                known_provider_id_source = list(provider_catalog.keys()) or known_provider_id_source
        except Exception:
            logger.warning("provider_catalog_unavailable_for_canonicalization")
    known_provider_ids = build_known_provider_ids(known_provider_id_source)
    # The generate pipeline supports a broader set of runtime providers than
    # the current registry catalog may expose (e.g., env-only providers).
    known_provider_ids.update(
        {
            "aliyun",
            "azure_openai",
            "ollama_gcp",
            "llamacpp_gcp",
            "gemini",
            "deepseek",
            "openrouter",
            "openai",
            "anthropic",
            "groq",
            "siliconeflow",
        }
    )

    normalized_forced_provider = _normalize_provider_id(
        forced_provider,
        known_provider_ids=known_provider_ids,
    )
    normalized_request_provider = _normalize_provider_id(
        request.provider,
        known_provider_ids=known_provider_ids,
    )

    if forced_provider and normalized_forced_provider not in known_provider_ids:
        _raise_unknown_provider_selection(str(forced_provider), correlation_id)

    if request.provider and normalized_request_provider not in known_provider_ids:
        _raise_unknown_provider_selection(str(request.provider), correlation_id)

    provider_hint = normalized_forced_provider or normalized_request_provider
    forced_provider = normalized_forced_provider if forced_provider else None
    explicit_provider_selection = bool(
        provider_hint and (forced_provider is not None or request.provider is not None)
    )
    effective_forced_provider = (
        forced_provider or provider_hint if explicit_provider_selection else forced_provider
    )

    explicit_model_for_validation = (forced_model or "").strip()
    if (
        not explicit_model_for_validation
        and request.provider
        and request.model
        and str(request.model).strip()
        and str(request.model).strip() != "llama2"
    ):
        explicit_model_for_validation = str(request.model).strip()

    provider_meta = provider_index.get(provider_hint) if provider_hint else None

    if provider_hint and explicit_model_for_validation and provider_meta is not None:
        configured_models = set(provider_meta.get("models", set()))
        if explicit_model_for_validation not in configured_models:
            _raise_unknown_model_selection(
                provider=provider_hint,
                model=explicit_model_for_validation,
                correlation_id=correlation_id,
            )

    if provider_hint:
        if provider_meta and not bool(provider_meta.get("is_selectable", True)):
            if not explicit_provider_selection:
                _raise_unavailable_provider_selection(
                    provider=provider_hint,
                    health=str(provider_meta.get("health") or "unknown"),
                    health_reason=provider_meta.get("health_reason"),
                    correlation_id=correlation_id,
                )
            logger.info(
                "provider_hint_forced_despite_unselectable_catalog_health",
                extra={
                    "correlation_id": correlation_id,
                    "provider": provider_hint,
                    "catalog_health": str(provider_meta.get("health") or "unknown"),
                },
            )

    if explicit_provider_selection:
        explicit_block_reason = _provider_block_reason(
            provider_hint,
            provider_hint,
            forced_provider,
        )
        if explicit_block_reason is not None:
            if explicit_block_reason == "recently_unhealthy":
                logger.info(
                    "provider_block_ignored_for_explicit_selection",
                    extra={
                        "correlation_id": correlation_id,
                        "provider": provider_hint,
                        "block_reason": explicit_block_reason,
                    },
                )
            else:
                health, reason = _health_for_block_reason(explicit_block_reason)
                _raise_unavailable_provider_selection(
                    provider=provider_hint,
                    health=health,
                    health_reason=reason,
                    correlation_id=correlation_id,
                )

    simple_prompt = _is_simple_prompt(messages)
    hard_timeout_s = 20.0 if effective_forced_provider else (8.0 if simple_prompt else 20.0)
    request_deadline = time.time() + hard_timeout_s
    errors: List[str] = []

    context = GenerateContext(
        correlation_id=correlation_id,
        messages=messages,
        prompt=last_user,
        model=model,
        provider_hint=provider_hint,
        req_max_tokens=req_max_tokens,
        req_temperature=req_temperature,
        simple_prompt=simple_prompt,
        request_deadline=request_deadline,
        forced_provider=effective_forced_provider,
        forced_model=forced_model,
    )

    provider_order, provider_timeout_profiles = _build_provider_strategy(
        simple_prompt, effective_forced_provider
    )

    provider_config = load_provider_config()
    provider_context = ProviderContext(
        messages=context.messages,
        prompt=context.prompt,
        model=context.model,
        provider_hint=context.provider_hint,
        req_max_tokens=context.req_max_tokens,
        req_temperature=context.req_temperature,
        forced_model=context.forced_model,
        forced_provider=context.forced_provider,
        request_deadline=context.request_deadline,
    )

    logger.info(
        "generate_request_profile",
        extra={
            "correlation_id": correlation_id,
            "provider_hint": provider_hint or "",
            "forced_provider": effective_forced_provider or "",
            "model": model,
            "max_tokens_effective": req_max_tokens,
            "temperature_effective": req_temperature,
            "simple_prompt": simple_prompt,
        },
    )

    provider_attempts = build_provider_attempts(
        provider_context,
        provider_config,
        lambda name: _provider_timeout(
            name,
            context.request_deadline,
            provider_timeout_profiles,
            effective_forced_provider,
        ),
    )

    provider_labels = {
        "ollama_gcp": "Ollama/GCP",
        "llamacpp_gcp": "LlamaCpp/GCP",
        "aliyun": "Aliyun",
        "azure_openai": "Azure OpenAI",
        "gemini": "Gemini",
        "groq": "Groq",
        "deepseek": "DeepSeek",
        "siliconeflow": "SiliconeFlow",
        "openrouter": "OpenRouter",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
    }

    for provider_name in provider_order:
        if provider_name not in provider_attempts:
            logger.info(
                "provider_candidate_skipped",
                extra={
                    "correlation_id": correlation_id,
                    "provider": provider_name,
                    "reason": "not_in_provider_attempts",
                },
            )
            continue
        if not _provider_is_configured(provider_name, provider_config):
            logger.info(
                "provider_candidate_skipped",
                extra={
                    "correlation_id": correlation_id,
                    "provider": provider_name,
                    "reason": "not_configured",
                },
            )
            continue
        block_reason = _provider_block_reason(
            provider_name, provider_hint, effective_forced_provider
        )
        if block_reason is not None:
            if (
                explicit_provider_selection
                and provider_name == provider_hint
                and block_reason == "recently_unhealthy"
            ):
                logger.info(
                    "provider_block_ignored_for_explicit_selection",
                    extra={
                        "correlation_id": correlation_id,
                        "provider": provider_name,
                        "block_reason": block_reason,
                    },
                )
            else:
                logger.info(
                    "provider_candidate_skipped",
                    extra={
                        "correlation_id": correlation_id,
                        "provider": provider_name,
                        "reason": block_reason,
                    },
                )
                continue

        max_attempts = _max_attempts_for_provider(
            provider_name, simple_prompt, effective_forced_provider
        )
        logger.info(
            "provider_candidate_attempting",
            extra={
                "correlation_id": correlation_id,
                "provider": provider_name,
                "max_attempts": max_attempts,
            },
        )

        result = await _attempt_provider(
            provider_name=provider_name,
            provider_call=provider_attempts[provider_name],
            max_attempts=max_attempts,
            request_deadline=context.request_deadline,
            errors=errors,
            provider_labels=provider_labels,
            correlation_id=correlation_id,
        )
        if result is not None:
            logger.info(
                "provider_candidate_selected",
                extra={
                    "correlation_id": correlation_id,
                    "provider": str(result.get("provider") or provider_name),
                    "model": str(result.get("model") or model),
                    "finish_reason": str(result.get("finish_reason") or ""),
                    "max_tokens_effective": req_max_tokens,
                },
            )
            return result

    if _GENERATE_OUTAGE_FALLBACK and not explicit_provider_selection:
        fallback_content = _build_outage_fallback_text(context.prompt)
        logger.warning(
            "Returning outage fallback response",
            extra={
                "correlation_id": correlation_id,
                "provider_errors": errors,
            },
        )
        return {
            "content": fallback_content,
            "response": fallback_content,
            "model": "outage-fallback",
            "provider": "fallback_unavailable",
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
            "finish_reason": "stop",
            "correlation_id": correlation_id,
        }

    if explicit_provider_selection:
        logger.error(
            "Explicit provider request failed",
            extra={
                "correlation_id": correlation_id,
                "provider": provider_hint,
                "errors": errors,
            },
        )
        raise_problem(
            status=503,
            title="Service Unavailable",
            detail="Selected inference provider unavailable.",
            type_uri="https://goblin-backend.onrender.com/errors/service-unavailable",
            code=ErrorCodes.SERVICE_UNAVAILABLE,
            instance=correlation_id,
            headers={"X-Correlation-ID": correlation_id},
        )

    logger.error(
        "All providers failed",
        extra={"correlation_id": correlation_id, "provider_errors": errors},
    )
    raise_problem(
        status=503,
        title="Service Unavailable",
        detail="All inference providers unavailable.",
        type_uri="https://goblin-backend.onrender.com/errors/service-unavailable",
        code=ErrorCodes.SERVICE_UNAVAILABLE,
        instance=correlation_id,
        headers={"X-Correlation-ID": correlation_id},
    )
