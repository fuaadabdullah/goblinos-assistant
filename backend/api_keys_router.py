from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import json

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class ApiKeyRequest(BaseModel):
    key: str


class ApiKeyResponse(BaseModel):
    key: Optional[str] = None
    provider: str


# Simple file-based storage for API keys (in production, use proper secrets management)
API_KEYS_FILE = os.path.join(os.path.dirname(__file__), "api_keys.json")


def load_api_keys():
    """Load API keys from file"""
    if os.path.exists(API_KEYS_FILE):
        try:
            with open(API_KEYS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_api_keys(keys):
    """Save API keys to file"""
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)


@router.post("/{provider}")
async def store_api_key(provider: str, request: ApiKeyRequest):
    """Store an API key for a provider"""
    try:
        keys = load_api_keys()
        keys[provider] = request.key
        save_api_keys(keys)
        return {"message": f"API key stored for {provider}"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to store API key: {str(e)}"
        )


@router.get("/{provider}", response_model=ApiKeyResponse)
async def get_api_key(provider: str):
    """Get an API key for a provider"""
    try:
        keys = load_api_keys()
        key = keys.get(provider)
        return ApiKeyResponse(key=key, provider=provider)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve API key: {str(e)}"
        )


@router.delete("/{provider}")
async def delete_api_key(provider: str):
    """Delete an API key for a provider"""
    try:
        keys = load_api_keys()
        if provider in keys:
            del keys[provider]
            save_api_keys(keys)
        return {"message": f"API key deleted for {provider}"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete API key: {str(e)}"
        )
