from __future__ import annotations

import os
from hmac import compare_digest

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..auth.policies import AuthScope
from ..services.imports import get_auth_service

security = HTTPBearer()
security_dependency = Security(security)


def require_scope(required_scope: AuthScope):
    """Dependency to require a specific scope."""

    def scope_checker(
        credentials: HTTPAuthorizationCredentials = security_dependency,
    ) -> list[str]:
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


def _get_expected_internal_proxy_key() -> str:
    for env_name in ("INTERNAL_PROXY_API_KEY", "BACKEND_API_KEY", "INTERNAL_API_SECRET"):
        value = (os.getenv(env_name) or "").strip()
        if value:
            return value
    return ""


def require_internal_proxy_key(request: Request) -> None:
    """
    Validate the internal Next.js -> FastAPI proxy key.

    If no key is configured, this check is a no-op for local development.
    """
    expected = _get_expected_internal_proxy_key()
    if not expected:
        return

    provided = (request.headers.get("x-internal-api-key") or "").strip()
    if not provided or not compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid internal proxy key")


__all__ = ["require_scope", "require_internal_proxy_key"]
