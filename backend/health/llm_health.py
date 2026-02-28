import os
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter

router = APIRouter()


def _normalize_health_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return f"{url}/health"


def _resolve_gateway_url() -> Optional[str]:
    explicit = os.getenv("LLM_GATEWAY_HEALTH_URL")
    if explicit:
        return explicit

    candidates = [
        os.getenv("LLM_GATEWAY_URL"),
        os.getenv("MODEL_GATEWAY_URL"),
        os.getenv("LOCAL_LLM_PROXY_URL"),
        os.getenv("KAMATERA_LLM_URL"),
        os.getenv("OLLAMA_BASE_URL"),
    ]

    for candidate in candidates:
        if candidate:
            return _normalize_health_url(candidate)

    return None


def _build_auth_header() -> Dict[str, str]:
    token = (
        os.getenv("LLM_GATEWAY_AUTH_TOKEN")
        or os.getenv("LLM_GATEWAY_JWT")
        or os.getenv("MODEL_GATEWAY_JWT")
        or os.getenv("LOCAL_LLM_API_KEY")
    )
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


@router.get("/health/llm")
async def llm_health() -> Dict[str, Any]:
    gateway_url = _resolve_gateway_url()
    if not gateway_url:
        return {
            "status": "unconfigured",
            "detail": "No LLM gateway configured",
        }

    headers = _build_auth_header()
    timeout_seconds = float(os.getenv("LLM_GATEWAY_HEALTH_TIMEOUT", "3"))

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(gateway_url, headers=headers)
            latency_ms = int((time.perf_counter() - started) * 1000)
            payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else None

            return {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "gateway_url": gateway_url,
                "http_status": response.status_code,
                "latency_ms": latency_ms,
                "response": payload,
            }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "status": "unhealthy",
            "gateway_url": gateway_url,
            "latency_ms": latency_ms,
            "error": str(exc),
        }
