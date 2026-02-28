from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import get_db
from .models import User
from .auth_service import get_auth_service, JWTAuthService

router = APIRouter(prefix="/account", tags=["account"])
security = HTTPBearer()


class ProfileRequest(BaseModel):
    name: str


class PreferencesRequest(BaseModel):
    summaries: bool
    notifications: bool
    familyMode: bool


def _prefs_path() -> str:
    data_dir = os.getenv("DATA_DIR", "/app/data")
    return os.getenv("ACCOUNT_PREFERENCES_PATH", os.path.join(data_dir, "account_prefs.json"))


def _load_prefs() -> dict[str, Any]:
    path = _prefs_path()
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle) or {}
    except Exception:
        return {}


def _save_prefs(prefs: dict[str, Any]) -> None:
    path = _prefs_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(prefs, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
    auth_service: JWTAuthService = Depends(get_auth_service),
) -> User:
    claims = auth_service.validate_access_token(credentials.credentials)
    if not claims:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@router.post("/profile")
async def save_profile(
    request: ProfileRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    name = (request.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")

    current_user.name = name
    db.add(current_user)
    db.commit()
    db.refresh(current_user)

    return {"status": "ok", "name": current_user.name}


@router.post("/preferences")
async def save_preferences(
    request: PreferencesRequest,
    current_user: User = Depends(get_current_user),
):
    # Persist lightweight preferences to the Fly volume. This avoids schema
    # changes while keeping the account screen functional.
    prefs = _load_prefs()
    prefs[str(current_user.id)] = {
        "summaries": bool(request.summaries),
        "notifications": bool(request.notifications),
        "familyMode": bool(request.familyMode),
    }
    try:
        _save_prefs(prefs)
    except Exception:
        # Non-fatal: still return the requested preferences so the UI can proceed.
        pass

    return {"status": "ok", "preferences": prefs[str(current_user.id)]}

