from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from .database import get_db
from .models import ProviderCredential, User
from .models_base import (
    AuditLog,
    Stream,
    StreamChunk,
    SupportMessage,
    Task,
    UserRoleAssignment,
    UserSession,
)
from .auth_service import get_auth_service, JWTAuthService

logger = logging.getLogger(__name__)

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


@router.delete("")
async def delete_account(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete the authenticated user's account and all of their data.

    Purges, in dependency-safe order: sessions, role assignments, audit
    entries, tasks, streams (+ chunks), support messages, provider
    credentials created by the user, chat history, stored preferences,
    and finally the user record itself.

    Not deleted here (documented limitations):
    - `routing_requests`: rows carry no user id, so they cannot be
      attributed to a single user; they age out via the retention purge
      (see docs/privacy.md).
    - API keys in the admin `api_key_store`: admin-scoped with no user
      linkage; only provider credentials with `created_by == <user id>`
      are removed.
    """
    uid = current_user.id
    uid_str = str(uid)
    deleted: dict[str, int] = {}

    def _bulk_delete(model, *criteria) -> int:
        """Delete rows matching criteria; never raise (best-effort per table)."""
        try:
            count = db.query(model).filter(*criteria).delete(synchronize_session=False)
            db.commit()
            return int(count or 0)
        except Exception:
            db.rollback()
            logger.warning(
                "Account deletion: could not purge %s for user %s",
                getattr(model, "__tablename__", model),
                uid_str,
                exc_info=True,
            )
            return 0

    # FK dependents first, the app_users row last.
    deleted["user_sessions"] = _bulk_delete(UserSession, UserSession.user_id == uid)
    deleted["user_role_assignments"] = _bulk_delete(
        UserRoleAssignment,
        or_(UserRoleAssignment.user_id == uid, UserRoleAssignment.assigned_by == uid),
    )
    deleted["audit_logs"] = _bulk_delete(AuditLog, AuditLog.actor_id == uid)
    deleted["tasks"] = _bulk_delete(Task, Task.user_id == uid)
    deleted["stream_chunks"] = _bulk_delete(
        StreamChunk,
        StreamChunk.stream_id.in_(
            db.query(Stream.id).filter(Stream.user_id == uid)
        ),
    )
    deleted["streams"] = _bulk_delete(Stream, Stream.user_id == uid)
    deleted["support_messages"] = _bulk_delete(
        SupportMessage, SupportMessage.user_id == uid
    )
    # ProviderCredential.created_by is "User ID or 'system'" (string).
    deleted["provider_credentials"] = _bulk_delete(
        ProviderCredential, ProviderCredential.created_by == uid_str
    )

    # Chat history table has no ORM model; best-effort raw SQL.
    try:
        result = db.execute(
            text("DELETE FROM chat_messages WHERE user_id = :uid"),
            {"uid": uid_str},
        )
        db.commit()
        deleted["chat_messages"] = int(result.rowcount or 0)
    except Exception:
        db.rollback()
        logger.warning(
            "Account deletion: could not purge chat_messages for user %s",
            uid_str,
            exc_info=True,
        )
        deleted["chat_messages"] = 0

    # Stored account preferences (server-side JSON, keyed by user id).
    try:
        prefs = _load_prefs()
        if uid_str in prefs:
            del prefs[uid_str]
            _save_prefs(prefs)
            deleted["preferences"] = 1
        else:
            deleted["preferences"] = 0
    except Exception:
        logger.warning(
            "Account deletion: could not purge preferences for user %s",
            uid_str,
            exc_info=True,
        )
        deleted["preferences"] = 0

    # The user record itself, last.
    try:
        db.query(User).filter(User.id == uid).delete(synchronize_session=False)
        db.commit()
        deleted["user"] = 1
    except Exception:
        db.rollback()
        logger.error(
            "Account deletion: could not delete user row for %s", uid_str, exc_info=True
        )
        raise HTTPException(
            status_code=409,
            detail="Account deletion incomplete: user record could not be removed",
        )

    return {"status": "deleted", "user_id": uid_str, "deleted": deleted}

