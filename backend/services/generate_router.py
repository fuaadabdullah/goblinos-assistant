"""Generate completion API router.

Exposes /generate and /models endpoints for inference request handling.
All provider orchestration and fallback logic is delegated to generate_service.
"""

from urllib.parse import quote, unquote
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, Response

from .generate_service import generate_completion
from .generate_models import GenerateRequest
from ..providers.registry import get_provider_registry

# Create separate routers for /v1/api and /v1/models endpoints
# (Note: these will be included under v1_router with /v1 prefix in main.py)
ollama_router = APIRouter(prefix="/api", tags=["generate"])
models_router = APIRouter(prefix="/models", tags=["models"])


@ollama_router.post("/generate")
async def ollama_generate(request: GenerateRequest, req: Request, response: Response):
    """Generate completion using default provider routing.

    Args:
        request: GenerateRequest with messages, optional model/provider hints
        req: FastAPI Request object for extracting correlation ID
        response: FastAPI Response object for setting headers

    Returns:
        Dict with generated content, usage, provider, and correlation ID
    """
    return await generate_completion(
        request=request,
        correlation_id=getattr(req.state, "request_id", None),
        response=response,
    )


@models_router.get("/routes")
async def list_model_routes():
    """List all available model routes across configured providers.

    Returns a catalog of provider+model combinations with their endpoints.
    """
    catalog = get_provider_registry().get_provider_catalog()
    routes: List[Dict[str, Any]] = []
    for provider_id, item in sorted(catalog.items()):
        models = item.get("models") or []
        if not isinstance(models, list):
            continue
        for model_name in models:
            if not isinstance(model_name, str) or not model_name:
                continue
            routes.append(
                {
                    "provider": provider_id,
                    "model": model_name,
                    "endpoint": f"/v1/models/{provider_id}/{quote(model_name, safe='')}/chat",
                }
            )
    return {"routes": routes, "count": len(routes)}


@models_router.post("/{provider}/{model}/chat")
async def chat_for_model(
    provider: str,
    model: str,
    request: GenerateRequest,
    req: Request,
    response: Response,
):
    """Generate completion for a specific provider+model combination.

    Args:
        provider: Provider ID (e.g. 'openai', 'ollama_gcp', 'gemini')
        model: Model name (URL-encoded)
        request: GenerateRequest with messages
        req: FastAPI Request for correlation ID
        response: FastAPI Response for headers

    Returns:
        Dict with generated completion, usage, and metadata

    Raises:
        HTTPException 404: If provider or model not found/configured
    """
    provider_id = provider.strip()
    decoded_model = unquote(model).strip()
    catalog = get_provider_registry().get_provider_catalog()
    provider_meta = catalog.get(provider_id) or catalog.get(provider)
    if not provider_meta:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    configured_models = provider_meta.get("models") or []
    if decoded_model not in configured_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{decoded_model}' not configured for provider '{provider_id}'",
        )
    return await generate_completion(
        request=request,
        forced_provider=provider_id,
        forced_model=decoded_model,
        correlation_id=getattr(req.state, "request_id", None),
        response=response,
    )
