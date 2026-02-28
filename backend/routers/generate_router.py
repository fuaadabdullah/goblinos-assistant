from typing import Any
from urllib.parse import quote, unquote

from fastapi import APIRouter, HTTPException, Request, Response

from ..providers.registry import get_provider_registry
from ..services.generate_service import GenerateRequest, generate_completion

ollama_router = APIRouter(prefix="/v1/api", tags=["api"])
models_router = APIRouter(prefix="/v1/models", tags=["models"])


@ollama_router.post("/generate")
async def ollama_generate(request: GenerateRequest, req: Request, response: Response):
    return await generate_completion(
        request=request,
        correlation_id=getattr(req.state, "request_id", None),
        response=response,
    )


@models_router.get("/routes")
async def list_model_routes():
    catalog = get_provider_registry().get_provider_catalog()
    routes: list[dict[str, Any]] = []
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


__all__ = ["ollama_router", "models_router"]
