"""
API key management router using encrypted storage.

Replaces the file-based api_keys_router.py with secure encrypted storage.
"""

from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os

from .api_key_store import APIKeyStore
from .policies import AuthScope
from ..auth_service import get_auth_service

router = APIRouter(prefix="/api-keys", tags=["api-keys"])

# Initialize API key store
API_KEY_STORE_PATH = os.path.join(os.path.dirname(__file__), "api_keys.enc")
api_key_store = APIKeyStore(API_KEY_STORE_PATH)

# Security schemes
security = HTTPBearer()


class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str
    scopes: List[str]
    expires_in_days: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class APIKeyResponse(BaseModel):
    """API key response."""

    key_id: str
    name: str
    scopes: List[str]
    created_at: str
    expires_at: Optional[str] = None
    last_used: Optional[str] = None
    active: bool


class CreateAPIKeyResponse(BaseModel):
    """Response when creating a new API key."""

    key_id: str
    key: str  # The actual API key (only returned on creation)


def get_current_user_scopes(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> List[str]:
    """Get current user's scopes from JWT token."""
    auth_service = get_auth_service()

    # Try JWT token first
    token = credentials.credentials
    claims = auth_service.validate_access_token(token)

    if claims:
        # Convert AuthScope enums back to strings
        scopes = auth_service.get_user_scopes(claims)
        return [scope.value for scope in scopes]

    # Try API key
    api_key = api_key_store.verify_key(token)
    if api_key:
        return api_key.scopes

    raise HTTPException(status_code=401, detail="Invalid authentication")


def require_scope(required_scope: AuthScope):
    """Dependency to require a specific scope."""

    def scope_checker(scopes: List[str] = Depends(get_current_user_scopes)):
        if required_scope.value not in scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required scope: {required_scope.value}",
            )
        return scopes

    return scope_checker


@router.post("/", response_model=CreateAPIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    scopes: List[str] = Depends(require_scope(AuthScope.ADMIN_SYSTEM)),
):
    """Create a new API key."""
    try:
        key_id, actual_key = api_key_store.create_key(
            name=request.name,
            scopes=request.scopes,
            expires_in_days=request.expires_in_days,
            metadata=request.metadata,
        )

        return CreateAPIKeyResponse(key_id=key_id, key=actual_key)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create API key: {str(e)}"
        )


@router.get("/", response_model=List[APIKeyResponse])
async def list_api_keys(
    include_inactive: bool = False,
    scopes: List[str] = Depends(require_scope(AuthScope.ADMIN_SYSTEM)),
):
    """List all API keys."""
    try:
        keys = api_key_store.list_keys(include_inactive=include_inactive)
        return [
            APIKeyResponse(
                key_id=key.key_id,
                name=key.name,
                scopes=key.scopes,
                created_at=key.created_at.isoformat(),
                expires_at=key.expires_at.isoformat() if key.expires_at else None,
                last_used=key.last_used.isoformat() if key.last_used else None,
                active=key.active,
            )
            for key in keys
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list API keys: {str(e)}"
        )


@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str, scopes: List[str] = Depends(require_scope(AuthScope.ADMIN_SYSTEM))
):
    """Get details of a specific API key."""
    try:
        key = api_key_store.get_key(key_id)
        if not key:
            raise HTTPException(status_code=404, detail="API key not found")

        return APIKeyResponse(
            key_id=key.key_id,
            name=key.name,
            scopes=key.scopes,
            created_at=key.created_at.isoformat(),
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            last_used=key.last_used.isoformat() if key.last_used else None,
            active=key.active,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get API key: {str(e)}")


@router.delete("/{key_id}")
async def delete_api_key(
    key_id: str, scopes: List[str] = Depends(require_scope(AuthScope.ADMIN_SYSTEM))
):
    """Delete an API key permanently."""
    try:
        success = api_key_store.delete_key(key_id)
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")

        return {"message": f"API key {key_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete API key: {str(e)}"
        )


@router.post("/{key_id}/revoke")
async def revoke_api_key(
    key_id: str, scopes: List[str] = Depends(require_scope(AuthScope.ADMIN_SYSTEM))
):
    """Revoke an API key (deactivate without deleting)."""
    try:
        success = api_key_store.revoke_key(key_id)
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")

        return {"message": f"API key {key_id} revoked successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to revoke API key: {str(e)}"
        )


@router.post("/{key_id}/rotate", response_model=CreateAPIKeyResponse)
async def rotate_api_key(
    key_id: str, scopes: List[str] = Depends(require_scope(AuthScope.ADMIN_SYSTEM))
):
    """Rotate an API key (generate new key with same permissions)."""
    try:
        result = api_key_store.rotate_key(key_id)
        if not result:
            raise HTTPException(status_code=404, detail="API key not found")

        new_key_id, new_key = result
        return CreateAPIKeyResponse(key_id=new_key_id, key=new_key)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to rotate API key: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_expired_keys(
    scopes: List[str] = Depends(require_scope(AuthScope.ADMIN_SYSTEM)),
):
    """Clean up expired API keys."""
    try:
        cleaned_count = api_key_store.cleanup_expired_keys()
        return {"message": f"Cleaned up {cleaned_count} expired API keys"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup keys: {str(e)}")


# Legacy provider-specific endpoints for backward compatibility
class ProviderAPIKeyRequest(BaseModel):
    """Request for provider API key (legacy)."""

    key: str


class ProviderAPIKeyResponse(BaseModel):
    """Response for provider API key (legacy)."""

    key: Optional[str] = None
    provider: str


@router.post("/providers/{provider}")
async def store_provider_api_key(
    provider: str,
    request: ProviderAPIKeyRequest,
    scopes: List[str] = Depends(require_scope(AuthScope.ADMIN_SYSTEM)),
):
    """Store an API key for a provider (legacy endpoint)."""
    try:
        # Create an API key with provider-specific scopes
        key_id, actual_key = api_key_store.create_key(
            name=f"provider-{provider}",
            scopes=["read:models", "write:conversations"],  # Basic provider scopes
            metadata={"provider": provider, "legacy": True},
        )

        return {"message": f"API key stored for provider {provider}", "key_id": key_id}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to store provider API key: {str(e)}"
        )


@router.get("/providers/{provider}", response_model=ProviderAPIKeyResponse)
async def get_provider_api_key(
    provider: str, scopes: List[str] = Depends(require_scope(AuthScope.ADMIN_SYSTEM))
):
    """Get an API key for a provider (legacy endpoint)."""
    try:
        # Find keys for this provider
        keys = api_key_store.list_keys()
        for key in keys:
            if key.metadata.get("provider") == provider and key.active:
                return ProviderAPIKeyResponse(key="***", provider=provider)

        return ProviderAPIKeyResponse(key=None, provider=provider)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve provider API key: {str(e)}"
        )


@router.delete("/providers/{provider}")
async def delete_provider_api_key(
    provider: str, scopes: List[str] = Depends(require_scope(AuthScope.ADMIN_SYSTEM))
):
    """Delete an API key for a provider (legacy endpoint)."""
    try:
        # Find and delete keys for this provider
        keys = api_key_store.list_keys(include_inactive=True)
        deleted_count = 0

        for key in keys:
            if key.metadata.get("provider") == provider:
                api_key_store.delete_key(key.key_id)
                deleted_count += 1

        return {"message": f"Deleted {deleted_count} API keys for provider {provider}"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete provider API key: {str(e)}"
        )
