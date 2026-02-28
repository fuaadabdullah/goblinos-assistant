"""
Best-effort warm-up for self-hosted GCP LLM endpoints (Ollama + llama.cpp).

Goals:
- Reduce cold-start latency for first user request.
- Never block request handling (callers should schedule in background).
- Never raise (best-effort).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_last_warm_at: float = 0.0
_warm_guard = threading.Lock()


def _is_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_falsy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "off"}


def _default_enabled() -> bool:
    # Default: enable when any self-hosted endpoints are configured.
    # Prefer explicit GCP URLs, but fall back to OLLAMA_BASE_URL for deployments
    # that only set the generic base URL.
    return bool(
        (os.getenv("OLLAMA_GCP_URL") or os.getenv("OLLAMA_BASE_URL") or "").strip()
        or (os.getenv("LLAMACPP_GCP_URL") or "").strip()
    )


def _parse_float_env(name: str, default: float, *, minimum: Optional[float] = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.debug("Invalid %s=%r, using default", name, raw)
        return default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _enabled() -> bool:
    raw = os.getenv("GCP_WARMUP_ENABLED")
    if raw is None or raw == "":
        return _default_enabled()
    if _is_falsy(raw):
        return False
    return _is_truthy(raw)


def _should_run(now: float) -> bool:
    interval_s = _parse_float_env("GCP_WARMUP_INTERVAL_S", 300.0, minimum=5.0)
    global _last_warm_at
    if (now - _last_warm_at) < interval_s:
        return False
    _last_warm_at = now
    return True


def _ollama_url() -> str:
    return (os.getenv("OLLAMA_GCP_URL") or os.getenv("OLLAMA_BASE_URL") or "").strip().rstrip("/")


def _llamacpp_url() -> str:
    return (os.getenv("LLAMACPP_GCP_URL") or "").strip().rstrip("/")


def _local_llm_key() -> str:
    return (os.getenv("LOCAL_LLM_API_KEY") or os.getenv("GCP_LLM_API_KEY") or "").strip()


def _ollama_warm_model() -> str:
    return (
        (os.getenv("OLLAMA_GCP_WARMUP_MODEL") or "").strip()
        or (os.getenv("OLLAMA_GCP_DEFAULT_MODEL") or "").strip()
        or "gemma:2b"
    )


def _llamacpp_warm_model() -> str:
    return (
        (os.getenv("LLAMACPP_GCP_WARMUP_MODEL") or "").strip()
        or (os.getenv("LLAMACPP_GCP_DEFAULT_MODEL") or "").strip()
        or "phi-3-mini-4k-instruct-q4"
    )


async def warm_gcp_endpoints(reason: Optional[str] = None) -> None:
    """
    Warm configured GCP endpoints.

    - Keepalive pings: cheap metadata endpoints.
    - Optional tiny inference call: primes model load / JIT.
    """
    if not _enabled():
        return

    now = time.monotonic()
    with _warm_guard:
        if not _should_run(now):
            return

    ollama_url = _ollama_url()
    llamacpp_url = _llamacpp_url()
    if not ollama_url and not llamacpp_url:
        return

    inference_enabled = os.getenv("GCP_WARMUP_INFERENCE_ENABLED")
    inference_on = (
        True
        if inference_enabled is None or inference_enabled == ""
        else _is_truthy(inference_enabled)
    )

    key = _local_llm_key()
    headers_ollama = {}
    headers_llamacpp = {"Content-Type": "application/json"}
    if key:
        headers_llamacpp["x-api-key"] = key
        headers_ollama["Authorization"] = f"Bearer {key}"
        headers_ollama["X-API-Key"] = key

    connect_timeout = _parse_float_env("GCP_WARMUP_CONNECT_TIMEOUT_S", 2.0, minimum=0.2)
    read_timeout = _parse_float_env("GCP_WARMUP_READ_TIMEOUT_S", 8.0, minimum=0.5)
    timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
    transport = httpx.AsyncHTTPTransport(retries=1)
    limits = httpx.Limits(max_keepalive_connections=2, max_connections=4)
    async with httpx.AsyncClient(timeout=timeout, transport=transport, limits=limits) as client:
        tasks = []

        async def _safe_get(name: str, url: str, headers: dict) -> None:
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code >= 400:
                    logger.debug(
                        "Warm-up keepalive non-2xx",
                        extra={"provider": name, "status_code": resp.status_code, "reason": reason},
                    )
            except Exception as exc:
                logger.debug(
                    "Warm-up keepalive failed",
                    extra={"provider": name, "error": type(exc).__name__, "reason": reason},
                )

        if ollama_url:
            tasks.append(_safe_get("ollama_gcp", f"{ollama_url}/api/tags", headers_ollama))
        if llamacpp_url:
            tasks.append(_safe_get("llamacpp_gcp", f"{llamacpp_url}/v1/models", headers_llamacpp))

        async def _ollama_tiny_infer() -> None:
            if not ollama_url:
                return
            try:
                await client.post(
                    f"{ollama_url}/api/generate",
                    headers={"Content-Type": "application/json", **headers_ollama},
                    json={
                        "model": _ollama_warm_model(),
                        "prompt": "ping",
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                )
            except Exception as exc:
                logger.debug(
                    "Warm-up ollama tiny inference failed",
                    extra={"provider": "ollama_gcp", "error": type(exc).__name__, "reason": reason},
                )

        async def _llamacpp_tiny_infer() -> None:
            if not llamacpp_url:
                return
            payload = {
                "model": _llamacpp_warm_model(),
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0.0,
            }
            try:
                for path in ("/v1/chat/completions", "/chat/completions"):
                    resp = await client.post(
                        f"{llamacpp_url}{path}",
                        headers=headers_llamacpp,
                        json=payload,
                    )
                    if resp.status_code in {404, 405}:
                        continue
                    return
            except Exception as exc:
                logger.debug(
                    "Warm-up llama.cpp tiny inference failed",
                    extra={
                        "provider": "llamacpp_gcp",
                        "error": type(exc).__name__,
                        "reason": reason,
                    },
                )

        if inference_on:
            tasks.append(_ollama_tiny_infer())
            tasks.append(_llamacpp_tiny_infer())

        # Use gather so warm-up is best-effort for each endpoint.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
