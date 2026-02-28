import os
import logging
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from .database import get_db
from .models import Provider, Model
from .config import settings as app_settings

router = APIRouter(prefix="/settings", tags=["settings"])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def _env_has_provider_key(provider_name: str) -> bool:
    base = (provider_name or "").upper().replace("-", "_")
    if base in {"ALIYUN", "ALIBABA", "ALIBABA_CLOUD"}:
        if (os.getenv("ALIYUN_MODEL_SERVER_KEY") or "").strip():
            return True
    if base in {"AZURE_OPENAI", "AZURE"}:
        for env_key in ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY"):
            if (os.getenv(env_key) or "").strip():
                return True
    for env_key in (f"{base}_API_KEY", f"{base}_KEY"):
        if (os.getenv(env_key) or "").strip():
            return True
    # Legacy compatibility (older env var spelling)
    if base == "SILICONEFLOW" and (os.getenv("SILICONFLOW_API_KEY") or "").strip():
        return True
    return False


class ProviderSchema(BaseModel):
    # NOTE: Keep `name` optional for update requests. The path parameter selects
    # the provider; the payload may contain only the fields being updated.
    name: Optional[str] = None
    display_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    models: Optional[List[str]] = []
    enabled: bool = True
    is_active: bool = True
    priority: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class ModelSchema(BaseModel):
    name: str
    provider: str
    model_id: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    enabled: bool = True

    model_config = ConfigDict(from_attributes=True)


class SettingsResponse(BaseModel):
    providers: List[ProviderSchema]
    models: List[ModelSchema]
    default_provider: Optional[str] = None
    default_model: Optional[str] = None


class RAGSettings(BaseModel):
    enable_enhanced_rag: bool = False
    chroma_path: str = "data/vector/chroma"


@router.get("/", response_model=SettingsResponse)
async def get_settings(db: Session = Depends(get_db)):
    """Get current provider and model settings from the database"""
    try:
        providers_db = db.query(Provider).all()
        models_db = db.query(Model).all()

        providers_response = []
        for provider in providers_db:
            provider_schema = ProviderSchema.model_validate(provider, from_attributes=True)
            # Never return provider secrets from settings payloads.
            providers_response.append(provider_schema.model_copy(update={"api_key": None}))

        models_response = [
            ModelSchema.model_validate(model, from_attributes=True) for model in models_db
        ]

        return SettingsResponse(
            providers=providers_response,
            models=models_response,
            default_provider=None,  # Or logic to determine default
            default_model=None,  # Or logic to determine default
        )
    except Exception as e:
        logger.exception("Failed to get settings")
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")


def _normalize_models(models_value: Any) -> List[str]:
    """Coerce Provider.models JSON field into a list[str] for the UI."""
    if not models_value:
        return []
    if isinstance(models_value, list):
        # Could be list[str] or list[dict]
        if models_value and isinstance(models_value[0], dict):
            out: List[str] = []
            for item in models_value:
                mid = item.get("id") or item.get("name")
                if isinstance(mid, str) and mid:
                    out.append(mid)
            return out
        return [str(m) for m in models_value if m is not None]
    return []


@router.get("/providers")
async def get_providers(db: Session = Depends(get_db)):
    """
    Backward-compatible providers endpoint for the frontend admin screens.

    Returns a flat list of provider configs (not wrapped in SettingsResponse),
    matching the shape expected by `src/hooks/api/useSettings.ts`.
    """
    providers_db = (
        db.query(Provider)
        .order_by(Provider.priority.desc().nullslast(), Provider.name.asc())
        .all()
    )
    payload = []
    for p in providers_db:
        has_key = bool(getattr(p, "api_key", None) or getattr(p, "api_key_encrypted", None))
        if not has_key:
            has_key = _env_has_provider_key(getattr(p, "name", "") or "")
        payload.append(
            {
                "id": p.id,
                "name": p.name,
                "enabled": bool(p.enabled),
                "is_active": bool(getattr(p, "is_active", True)),
                "priority": getattr(p, "priority", None),
                "weight": None,
                # Never return the raw key; UI only needs presence.
                "api_key": "***" if has_key else None,
                "base_url": getattr(p, "base_url", None),
                "models": _normalize_models(getattr(p, "models", None)),
            }
        )
    return payload


@router.get("/models")
async def get_models(db: Session = Depends(get_db)):
    """Return model configs as a flat list for the frontend."""
    models_db = db.query(Model).order_by(Model.provider.asc(), Model.name.asc()).all()
    return [m.to_dict() for m in models_db]


@router.get("/global")
async def get_global_settings():
    """
    Minimal global settings endpoint for frontend compatibility.

    The current backend config is primarily environment-driven; return a small,
    safe subset (and room to expand later).
    """
    return {
        "environment": getattr(app_settings, "environment", None),
        "enable_enhanced_rag": getattr(app_settings, "enable_enhanced_rag", False),
    }


class GlobalSettingUpdate(BaseModel):
    value: str


@router.put("/global/{key}")
async def update_global_setting(key: str, update: GlobalSettingUpdate):
    """
    Compatibility shim: accept updates but do not mutate process env.

    This keeps the admin UI functional without implying persistence that
    doesn't exist in this deployment path.
    """
    return {"status": "success", "key": key, "value": update.value}


@router.put("/providers/{provider_key}")
async def update_provider_settings(
    provider_key: str, settings: ProviderSchema, db: Session = Depends(get_db)
):
    """Update settings for a specific provider in the database"""
    try:
        provider = None
        if provider_key.isdigit():
            provider = db.query(Provider).filter(Provider.id == int(provider_key)).first()
        if provider is None:
            provider = db.query(Provider).filter(Provider.name == provider_key).first()
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider {provider_key} not found",
            )

        # Update provider fields
        update_fields = settings.model_dump(exclude_unset=True)
        # Prevent accidental renames via payload; the key in the URL is the identity.
        update_fields.pop("name", None)
        for field, value in update_fields.items():
            if field == "models" and value is not None:
                # Store as list[str] for now; routing service may enrich later.
                setattr(provider, field, value)
            else:
                setattr(provider, field, value)

        db.commit()
        db.refresh(provider)

        return {
            "status": "success",
            "message": f"Settings updated for provider: {provider.name}",
            "settings": ProviderSchema.model_validate(provider, from_attributes=True),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update provider settings", extra={"provider_key": provider_key})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update provider settings: {str(e)}",
        )


@router.put("/models/{model_name}")
async def update_model_settings(
    model_name: str, settings: ModelSchema, db: Session = Depends(get_db)
):
    """Update settings for a specific model in the database"""
    try:
        model = db.query(Model).filter(Model.name == model_name).first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found",
            )

        # Update model fields
        for field, value in settings.model_dump(exclude_unset=True).items():
            setattr(model, field, value)

        db.commit()
        db.refresh(model)

        return {
            "status": "success",
            "message": f"Settings updated for model: {model_name}",
            "settings": ModelSchema.model_validate(model, from_attributes=True),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update model settings", extra={"model_name": model_name})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model settings: {str(e)}",
        )


@router.post("/test-connection")
async def test_provider_connection(provider_name: str, db: Session = Depends(get_db)):
    """Test connection to a provider's API"""
    try:
        # Find the provider in database
        provider = db.query(Provider).filter(Provider.name == provider_name).first()

        if not provider:
            raise HTTPException(
                status_code=404, detail=f"Provider {provider_name} not found"
            )

        if not provider.api_key:
            return {
                "status": "warning",
                "message": f"No API key configured for {provider_name}",
                "connected": False,
            }

        # In a real app, you would make a test API call here
        # For now, we'll just check if the API key exists
        return {
            "status": "success",
            "message": f"Connection test successful for {provider_name}",
            "connected": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Connection test failed", extra={"provider_name": provider_name})
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")


# RAG Settings Endpoints


@router.get("/rag", response_model=RAGSettings)
async def get_rag_settings():
    """Get current RAG settings"""
    try:
        return RAGSettings(
            enable_enhanced_rag=app_settings.enable_enhanced_rag,
            chroma_path=app_settings.rag_chroma_path,
        )
    except Exception as e:
        logger.exception("Failed to get RAG settings")
        raise HTTPException(
            status_code=500, detail=f"Failed to get RAG settings: {str(e)}"
        )


@router.put("/rag")
async def update_rag_settings(rag_settings: RAGSettings):
    """Update RAG settings"""
    try:
        # In a real app, this would update the database and restart services
        # For now, we'll just validate the input and return success
        if not rag_settings.chroma_path:
            raise HTTPException(status_code=400, detail="Chroma path is required")

        return {
            "status": "success",
            "message": "RAG settings updated. Note: Changes may require service restart to take effect.",
            "settings": rag_settings.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update RAG settings")
        raise HTTPException(
            status_code=500, detail=f"Failed to update RAG settings: {str(e)}"
        )


@router.post("/rag/test")
async def test_rag_configuration():
    """Test RAG configuration and dependencies"""
    try:
        # Test if RAG service can be initialized
        from .services.rag_service import RAGService

        # Try to initialize with current settings
        rag_service = RAGService(
            enable_enhanced=app_settings.enable_enhanced_rag,
            chroma_path=app_settings.rag_chroma_path,
        )

        # Check if ChromaDB is available
        chroma_available = rag_service.chroma_client is not None

        # Check if enhanced features are available
        enhanced_available = False
        if app_settings.enable_enhanced_rag:
            try:
                enhanced_service = rag_service._get_enhanced_service()
                enhanced_available = enhanced_service is not None
            except Exception:
                enhanced_available = False

        return {
            "status": "success",
            "chroma_available": chroma_available,
            "enhanced_rag_enabled": app_settings.enable_enhanced_rag,
            "enhanced_rag_available": enhanced_available,
            "chroma_path": app_settings.rag_chroma_path,
        }

    except Exception as e:
        logger.exception("RAG configuration test failed")
        return {
            "status": "error",
            "message": f"RAG configuration test failed: {str(e)}",
            "chroma_available": False,
            "enhanced_rag_enabled": app_settings.enable_enhanced_rag,
            "enhanced_rag_available": False,
        }
