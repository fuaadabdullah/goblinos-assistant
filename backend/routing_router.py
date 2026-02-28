"""
Routing router with real provider discovery, health monitoring, and intelligent task routing.
"""

from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime

from .database import get_db
from .services.routing import RoutingService
from .services.enhanced_routing import EnhancedRoutingService
from .services.imports import get_routing_service as get_unified_routing_service
from .providers.registry import get_provider_registry
from .auth.policies import AuthScope
from .auth_service import get_auth_service
from .schemas.v1.routing import EnhancedRouteRequest, ProviderInfo, RouteRequest
import os

# Get encryption key from environment
ROUTING_ENCRYPTION_KEY = os.getenv("ROUTING_ENCRYPTION_KEY", "default-dev-key-change-me")

# Initialize services
routing_service = None
enhanced_routing_service = None

router = APIRouter(prefix="/routing", tags=["routing"])


def get_routing_service(db: Session = Depends(get_db)) -> RoutingService:
    """Dependency to get routing service instance."""
    global routing_service
    if routing_service is None:
        routing_service = RoutingService(db, ROUTING_ENCRYPTION_KEY)
    return routing_service


def get_enhanced_routing_service(
    db: Session = Depends(get_db),
) -> EnhancedRoutingService:
    """Dependency to get enhanced routing service instance."""
    global enhanced_routing_service
    if enhanced_routing_service is None:
        enhanced_routing_service = EnhancedRoutingService(db, ROUTING_ENCRYPTION_KEY)
    return enhanced_routing_service

# Security schemes
security = HTTPBearer()


def require_scope(required_scope: AuthScope):
    """Dependency to require a specific scope."""

    def scope_checker(
        credentials: HTTPAuthorizationCredentials = Security(security),
    ) -> List[str]:
        auth_service = get_auth_service()

        # Try JWT token first
        token = credentials.credentials
        claims = auth_service.validate_access_token(token)

        if claims:
            # Convert AuthScope enums back to strings
            scopes = auth_service.get_user_scopes(claims)
            scope_values = [scope.value for scope in scopes]

            if required_scope.value not in scope_values:
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required scope: {required_scope.value}",
                )
            return scope_values

        raise HTTPException(status_code=401, detail="Invalid authentication")

    return scope_checker


def _normalize_models(models_value: Any) -> List[str]:
    """Coerce Provider.models JSON field into a list[str] for the UI."""
    if not models_value:
        return []
    if isinstance(models_value, list):
        if models_value and isinstance(models_value[0], dict):
            out: List[str] = []
            for item in models_value:
                mid = item.get("id") or item.get("name") or item.get("model")
                if isinstance(mid, str) and mid:
                    out.append(mid)
            return out
        return [str(m) for m in models_value if m is not None]
    return []


def _normalize_provider_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _discover_registry_providers() -> List[Dict[str, Any]]:
    """Discover providers from the provider registry catalog as a compatibility fallback."""
    try:
        catalog = get_provider_registry().get_provider_catalog()
    except Exception:
        return []

    providers: List[Dict[str, Any]] = []
    for raw_provider_id, meta in (catalog or {}).items():
        provider_name = _normalize_provider_name(raw_provider_id)
        if not provider_name:
            continue
        raw_models = meta.get("models", [])
        models = [
            {"id": model_id}
            for model_id in raw_models
            if isinstance(model_id, str) and model_id
        ]
        capabilities = meta.get("capabilities", [])
        if not isinstance(capabilities, list) or not capabilities:
            capabilities = ["chat"]
        providers.append(
            {
                "id": provider_name,
                "name": provider_name,
                "display_name": str(meta.get("display_name") or provider_name.replace("_", " ").title()),
                "capabilities": list(capabilities),
                "models": models,
                "priority": int(meta.get("priority_tier", 1) or 1),
                "is_active": bool(meta.get("is_active", True)),
            }
        )
    return providers


async def _discover_unified_providers(service: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    try:
        providers = await service.discover_providers()
    except Exception:
        providers = []

    for provider in providers or []:
        models = provider.get("models", [])
        if isinstance(models, list) and models and isinstance(models[0], str):
            models = [{"id": model_id} for model_id in models]
        provider_name = _normalize_provider_name(provider.get("name", provider.get("id", "")))
        if not provider_name:
            continue
        normalized.append(
            {
                "id": provider_name,
                "name": provider_name,
                "display_name": str(
                    provider.get(
                        "display_name",
                        provider_name.replace("_", " ").title(),
                    )
                ),
                "capabilities": list(provider.get("capabilities", []) or []),
                "models": models if isinstance(models, list) else [],
                "priority": int(provider.get("priority", 1) or 1),
                "is_active": bool(provider.get("is_active", False)),
            }
        )

    # Compat fallback: if unified discovery is sparse, merge in registry providers so
    # /v1/routing/providers* aligns with /v1/providers/models in production.
    merged: Dict[str, Dict[str, Any]] = {p["name"]: p for p in normalized if p.get("name")}
    registry_providers = _discover_registry_providers()
    for provider in registry_providers:
        provider_name = provider.get("name")
        if not provider_name:
            continue
        if provider_name in merged:
            existing = merged[provider_name]
            if not existing.get("models"):
                existing["models"] = provider.get("models", [])
            if not existing.get("capabilities"):
                existing["capabilities"] = provider.get("capabilities", [])
            continue
        merged[provider_name] = provider

    return sorted(merged.values(), key=lambda item: str(item.get("name", "")))


@router.get("/providers/details", response_model=List[ProviderInfo])
async def get_available_providers(
    service: Any = Depends(get_unified_routing_service),
):
    """Get list of all configured providers with their capabilities and status.

    Public read endpoint used by frontend provider bootstrap.
    """
    try:
        providers = await _discover_unified_providers(service)
        return [ProviderInfo(**provider) for provider in providers]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to discover providers: {str(e)}"
        )


@router.get("/providers/capability/{capability}", response_model=List[ProviderInfo])
async def get_providers_for_capability(
    capability: str,
    service: Any = Depends(get_unified_routing_service),
    scopes: List[str] = Depends(require_scope(AuthScope.READ_MODELS)),
):
    """Get providers that support a specific capability"""
    try:
        providers = await _discover_unified_providers(service)

        # Filter providers that support the capability
        suitable = []
        for provider in providers:
            if capability in provider["capabilities"]:
                suitable.append(ProviderInfo(**provider))

        return suitable
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get providers for capability: {str(e)}"
        )


@router.get("/providers", response_model=List[str])
async def list_provider_keys(service: Any = Depends(get_unified_routing_service)):
    """List enabled provider keys (frontend compatibility endpoint)."""
    providers = await _discover_unified_providers(service)
    return sorted({p.get("name", "") for p in providers if p.get("name")})


@router.get("/providers/{provider}", response_model=List[str])
async def list_provider_models(
    provider: str, service: Any = Depends(get_unified_routing_service)
):
    """List models for a provider (frontend compatibility endpoint)."""
    providers = await _discover_unified_providers(service)
    target = provider.replace("-", "_")
    for item in providers:
        candidate = str(item.get("name", ""))
        if candidate == provider or candidate == target:
            models = item.get("models", [])
            out: List[str] = []
            for model in models:
                if isinstance(model, dict):
                    mid = model.get("id")
                    if isinstance(mid, str) and mid:
                        out.append(mid)
                elif isinstance(model, str) and model:
                    out.append(model)
            return out
    return []


@router.get("/models", response_model=List[str])
async def list_available_models(service: Any = Depends(get_unified_routing_service)):
    """List all enabled model IDs across providers (frontend compatibility endpoint)."""
    model_ids: set[str] = set()
    providers = await _discover_unified_providers(service)
    for provider in providers:
        for model in provider.get("models", []):
            if isinstance(model, dict):
                mid = model.get("id")
                if isinstance(mid, str) and mid:
                    model_ids.add(mid)
            elif isinstance(model, str) and model:
                model_ids.add(model)

    return sorted(model_ids)


@router.get("/info")
async def routing_info(service: Any = Depends(get_unified_routing_service)):
    """Lightweight routing info for startup preflight (frontend compatibility endpoint)."""
    providers = await _discover_unified_providers(service)
    provider_count = len(providers)
    enabled_provider_count = sum(1 for p in providers if p.get("is_active"))
    model_ids: set[str] = set()
    for provider in providers:
        for model in provider.get("models", []):
            if isinstance(model, dict):
                mid = model.get("id")
                if isinstance(mid, str) and mid:
                    model_ids.add(mid)
            elif isinstance(model, str) and model:
                model_ids.add(model)
    model_count = len(model_ids)
    enabled_model_count = model_count

    return {
        "status": "ok",
        "providers": {"total": provider_count, "enabled": enabled_provider_count},
        "models": {"total": model_count, "enabled": enabled_model_count},
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/route")
async def route_request(
    request: RouteRequest,
    service: RoutingService = Depends(get_routing_service),
    scopes: List[str] = Depends(require_scope(AuthScope.WRITE_CONVERSATIONS)),
):
    """Route a request to the best available provider"""
    try:
        result = await service.route_request(
            capability=request.capability, requirements=request.requirements
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")


@router.post("/route/enhanced")
async def route_request_enhanced(
    request: EnhancedRouteRequest,
    service: EnhancedRoutingService = Depends(get_enhanced_routing_service),
    scopes: List[str] = Depends(require_scope(AuthScope.WRITE_CONVERSATIONS)),
):
    """
    Route a request using enhanced multi-factor decision algorithm.

    Includes advanced factors like time-of-day weighting, user tier prioritization,
    conversation context analysis, regional latency optimization, and more.
    """
    try:
        result = await service.select_provider_enhanced(
            capability=request.capability,
            requirements=request.requirements,
            sla_target_ms=request.sla_target_ms,
            cost_budget=request.cost_budget,
            latency_priority=request.latency_priority,
            user_region=request.user_region,
            conversation_history=request.conversation_history,
            request_content=request.request_content,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Enhanced routing failed: {str(e)}"
        )


@router.get("/health")
async def routing_health(
    service: RoutingService = Depends(get_routing_service),
    scopes: List[str] = Depends(require_scope(AuthScope.READ_USER)),
):
    """Check if the routing system is operational"""
    try:
        providers = await service.discover_providers()
        healthy_providers = [p for p in providers if p.get("is_active", False)]

        return {
            "status": "healthy" if healthy_providers else "degraded",
            "providers_available": len(providers),
            "healthy_providers": len(healthy_providers),
            "routing_system": "active",
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "routing_system": "failed"}
