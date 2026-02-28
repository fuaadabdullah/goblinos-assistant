from __future__ import annotations

import os
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import get_db
from .models import Provider
from .providers.registry import get_provider_registry
from .auth.dependencies import require_internal_proxy_key
from .services.encryption import EncryptionService
from .services.provider_catalog import build_models_payload

import logging

logger = logging.getLogger(__name__)

# Provider adapters (use DB-stored credentials when passed).
try:
    from .providers.ollama_adapter import OllamaAdapter  # type: ignore
except Exception:  # pragma: no cover
    OllamaAdapter = None  # type: ignore

try:
    from .providers.openai_adapter import OpenAIAdapter  # type: ignore
except Exception:  # pragma: no cover
    OpenAIAdapter = None  # type: ignore

try:
    from .providers.anthropic_adapter import AnthropicAdapter  # type: ignore
except Exception:  # pragma: no cover
    AnthropicAdapter = None  # type: ignore

try:
    from .providers.grok_adapter import GrokAdapter  # type: ignore
except Exception:  # pragma: no cover
    GrokAdapter = None  # type: ignore

try:
    from .providers.deepseek_adapter import DeepSeekAdapter  # type: ignore
except Exception:  # pragma: no cover
    DeepSeekAdapter = None  # type: ignore

try:
    from .providers.gemini_adapter import GeminiAdapter  # type: ignore
except Exception:  # pragma: no cover
    GeminiAdapter = None  # type: ignore

router = APIRouter(prefix="/providers", tags=["providers"])


class ProviderTestRequest(BaseModel):
    prompt: Optional[str] = None


class ProviderPriorityRequest(BaseModel):
    priority: int
    role: Optional[str] = None


class ProviderReorderRequest(BaseModel):
    providerIds: list[int]


def _resolve_provider_key(provider: Provider) -> str:
    # Prefer encrypted key if present; fall back to plaintext.
    encrypted = getattr(provider, "api_key_encrypted", None)
    if encrypted:
        try:
            return EncryptionService().decrypt(encrypted)
        except Exception:
            # Fall back to plaintext/env if decryption fails.
            pass

    key = getattr(provider, "api_key", None) or ""
    if key:
        return key
    # As a last resort, fall back to env var expected by registry.
    provider_name = (provider.name or "").strip().lower().replace("-", "_")
    env_candidates: list[str] = []
    if provider_name in {"aliyun", "alibaba", "alibaba_cloud"}:
        env_candidates.extend(
            ["ALIYUN_MODEL_SERVER_KEY", "ALIYUN_API_KEY", "ALIYUN_KEY"]
        )
    if provider_name in {"azure_openai", "azure"}:
        env_candidates.extend(
            ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY"]
        )
    base = provider_name.upper().replace("-", "_")
    env_candidates.extend([f"{base}_API_KEY", f"{base}_KEY"])

    seen: set[str] = set()
    for env_key in env_candidates:
        if env_key in seen:
            continue
        seen.add(env_key)
        value = os.getenv(env_key, "").strip()
        if value:
            return value
    return ""


def _adapter_for(name: str):
    n = (name or "").strip().lower().replace("-", "_")
    mapping = {
        "ollama": OllamaAdapter,
        "openai": OpenAIAdapter,
        "azure_openai": OpenAIAdapter,
        "azure": OpenAIAdapter,
        "aliyun": OpenAIAdapter,
        "alibaba": OpenAIAdapter,
        "alibaba_cloud": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "grok": GrokAdapter,
        "xai": GrokAdapter,
        "deepseek": DeepSeekAdapter,
        "gemini": GeminiAdapter,
        "google": GeminiAdapter,
    }
    adapter = mapping.get(n)
    return adapter if adapter is not None else None


def _pick_model(provider: Provider) -> Optional[str]:
    models = getattr(provider, "models", None)
    if not models:
        return None
    if isinstance(models, list) and models:
        first = models[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            mid = first.get("id") or first.get("name")
            if isinstance(mid, str) and mid:
                return mid
    return None


def _extract_provider_id(provider_obj: Any) -> Optional[str]:
    provider_id = getattr(provider_obj, "provider_id", None)
    if callable(provider_id):
        try:
            provider_id = provider_id()
        except Exception:
            provider_id = None
    if isinstance(provider_id, str) and provider_id:
        return provider_id
    return None


@router.get("/models")
async def list_provider_models(
    _internal_proxy_auth: None = Depends(require_internal_proxy_key),
):
    """
    Aggregate models from backend provider registry.

    Response shape is compatible with the Next.js `/api/models` contract.
    """
    registry = get_provider_registry()
    payload = build_models_payload(registry)
    logger.info(
        "providers_models_catalog_built",
        extra={
            "source": payload.get("source"),
            "provider_count": len(payload.get("providers") or []),
            "model_count": len(payload.get("models") or []),
        },
    )
    return payload


@router.post("/{provider_id}/test")
async def test_provider(
    provider_id: int,
    request: ProviderTestRequest | None = None,
    db: Session = Depends(get_db),
):
    """
    Frontend-compatible provider test endpoint.

    - No prompt: lightweight connectivity check (list models).
    - With prompt: run a tiny chat completion to validate inference.
    """
    provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    if not getattr(provider, "enabled", True):
        return {"success": False, "message": "Provider disabled", "latency": 0}

    adapter_class = _adapter_for(provider.name)
    if not adapter_class:
        return {
            "success": False,
            "message": f"No adapter available for provider '{provider.name}'",
            "latency": 0,
        }

    api_key = _resolve_provider_key(provider)
    base_url = getattr(provider, "base_url", None)

    if not api_key and provider.name.lower() not in {"ollama"}:
        return {
            "success": False,
            "message": f"No API key configured for provider '{provider.name}'",
            "latency": 0,
        }

    start = time.perf_counter()
    try:
        adapter = adapter_class(api_key, base_url)

        prompt = (request.prompt if request else None) or None
        if prompt:
            model = _pick_model(provider) or os.getenv("DEFAULT_TEST_MODEL") or ""
            if not model:
                # Fall back to listing models to pick one.
                models = await adapter.list_models()
                if models and isinstance(models[0], dict):
                    model = models[0].get("id") or ""
            if not model:
                raise RuntimeError("No model available to test")

            response = await adapter.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.0,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            return {
                "success": True,
                "message": "Prompt test succeeded",
                "latency": latency_ms,
                "response": response,
                "model_used": model,
            }

        # Connectivity test: list models (no inference cost).
        models = await adapter.list_models()
        latency_ms = int((time.perf_counter() - start) * 1000)
        if not models:
            return {
                "success": False,
                "message": "Provider reachable but returned no models",
                "latency": latency_ms,
            }
        return {
            "success": True,
            "message": f"Connection OK ({len(models)} models)",
            "latency": latency_ms,
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "success": False,
            "message": str(exc),
            "latency": latency_ms,
        }


@router.post("/{provider_id}/test-prompt")
async def test_provider_with_prompt(
    provider_id: int,
    request: ProviderTestRequest,
    db: Session = Depends(get_db),
):
    """Frontend-compatible alias for prompt-based provider tests."""
    return await test_provider(provider_id=provider_id, request=request, db=db)


@router.post("/{provider_id}/priority")
async def set_provider_priority(
    provider_id: int,
    request: ProviderPriorityRequest,
    db: Session = Depends(get_db),
):
    provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    provider.priority = int(request.priority)
    db.commit()
    db.refresh(provider)

    return {"success": True, "provider_id": provider.id, "priority": provider.priority}


@router.post("/reorder")
async def reorder_providers(
    request: ProviderReorderRequest,
    db: Session = Depends(get_db),
):
    """
    Persist drag-and-drop order by mapping it onto Provider.priority.

    Higher priority means more preferred; first item gets highest priority.
    """
    ids = request.providerIds or []
    if not ids:
        return {"success": True, "updated": 0}

    providers = db.query(Provider).filter(Provider.id.in_(ids)).all()
    by_id = {p.id: p for p in providers}

    # Assign descending priority based on order
    max_prio = len(ids)
    updated = 0
    for index, pid in enumerate(ids):
        provider = by_id.get(pid)
        if not provider:
            continue
        provider.priority = max_prio - index
        updated += 1

    db.commit()
    return {"success": True, "updated": updated}
