"""
Simplified JWT Authentication Service for Goblin Assistant

This module provides JWT token creation and validation for user authentication.
API keys are handled separately through the APIKeyStore.

Key Features:
- JWT token creation and validation
- Role-based access control with scopes
- Token expiration and refresh
- No complex session management
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Set
import jwt
from jwt import PyJWTError
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# Use try/except for flexible imports
try:
    from backend.auth.policies import (
        AuthScope,
        UserRole,
        get_scopes_for_role,
        validate_scopes,
    )
    from backend.auth.secrets_manager import get_secrets_manager
    from backend.database import get_db
    from backend.models import User
except ImportError:
    from .auth.policies import (
        AuthScope,
        UserRole,
        get_scopes_for_role,
        validate_scopes,
    )
    from .auth.secrets_manager import get_secrets_manager
    from .database import get_db
    from .models import User

load_dotenv()

# Configuration
JWT_ALGORITHM = "HS256"

# Token lifetimes
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Short-lived access tokens
REFRESH_TOKEN_EXPIRE_DAYS = 30  # Longer refresh tokens


class JWTAuthService:
    """Simplified JWT authentication service for user tokens."""

    def __init__(self):
        secrets_manager = get_secrets_manager()
        self.secret_key = secrets_manager.get_jwt_secret_key()
        self.algorithm = JWT_ALGORITHM

    def create_access_token(
        self,
        user_id: str,
        email: str,
        role: UserRole,
        additional_scopes: Optional[Set[AuthScope]] = None,
    ) -> str:
        """Create a JWT access token for a user.

        Args:
            user_id: User's unique identifier
            email: User's email address
            role: User's role
            additional_scopes: Additional scopes beyond role defaults

        Returns:
            JWT access token string
        """
        # Get base scopes from role
        scopes = get_scopes_for_role(role)

        # Add any additional scopes
        if additional_scopes:
            scopes.update(additional_scopes)

        # Ensure user_id is serializable (UUID -> str)
        user_id = str(user_id)
        # Create token payload
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        payload = {
            "sub": user_id,
            "email": email,
            "role": role.value,
            "scopes": [scope.value for scope in scopes],  # Store as strings
            "type": "access",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, user_id: str) -> str:
        """Create a JWT refresh token for a user.

        Args:
            user_id: User's unique identifier

        Returns:
            JWT refresh token string
        """
        # Ensure user_id is serializable
        user_id = str(user_id)
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        payload = {
            "sub": user_id,
            "type": "refresh",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a JWT token and return claims if valid.

        Args:
            token: JWT token string

        Returns:
            Token claims if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check token type and expiration
            token_type = payload.get("type")
            if token_type not in ["access", "refresh"]:
                return None

            # Convert scope strings back to AuthScope enums
            if "scopes" in payload:
                payload["scopes"] = validate_scopes(payload["scopes"])

            return payload

        except PyJWTError:
            return None

    def validate_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate an access token specifically.

        Args:
            token: JWT access token string

        Returns:
            Token claims if valid access token, None otherwise
        """
        payload = self.validate_token(token)
        if payload and payload.get("type") == "access":
            return payload
        return None

    def validate_refresh_token(self, token: str) -> Optional[str]:
        """Validate a refresh token and return user_id if valid.

        Args:
            token: JWT refresh token string

        Returns:
            User ID if valid refresh token, None otherwise
        """
        payload = self.validate_token(token)
        if payload and payload.get("type") == "refresh":
            return payload.get("sub")
        return None

    def refresh_access_token(self, refresh_token: str, db: Session) -> Optional[str]:
        """Create a new access token from a valid refresh token.

        Args:
            refresh_token: Valid refresh token
            db: Database session for user lookup

        Returns:
            New access token if refresh token is valid, None otherwise
        """
        payload = self.validate_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None

        user_id = payload.get("sub")
        if not user_id:
            return None
        if isinstance(user_id, str):
            try:
                import uuid

                user_id = uuid.UUID(user_id)
            except Exception:
                return None

        # Look up user from database
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None

        # Check if token version matches (for revocation support)
        token_version = payload.get("token_version")
        if token_version is not None and user.token_version != token_version:
            return None

        # Get scopes for user's role
        role = user.role
        if not isinstance(role, UserRole):
            try:
                role = UserRole(role)
            except Exception:
                role = UserRole.USER

        scopes = get_scopes_for_role(role)

        # Create new access token with proper user info
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_payload = {
            "sub": str(user.id),
            "type": "access",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "role": role.value,
            "scopes": [scope.value for scope in scopes],
            "token_version": user.token_version,
        }

        return jwt.encode(new_payload, self.secret_key, algorithm=self.algorithm)

    def get_user_scopes(self, token_claims: Dict[str, Any]) -> Set[AuthScope]:
        """Extract scopes from validated token claims.

        Args:
            token_claims: Validated token claims

        Returns:
            Set of AuthScope enums
        """
        scopes = token_claims.get("scopes", [])

        # If scopes are already AuthScope enums (from JWT decoding), return as set
        if isinstance(scopes, set) and all(
            isinstance(s, type(AuthScope.READ_USER)) for s in scopes
        ):
            return scopes

        # If scopes are strings, validate and convert
        if isinstance(scopes, list) and all(isinstance(s, str) for s in scopes):
            return validate_scopes(scopes)

        return set()

    def has_scope(self, token_claims: Dict[str, Any], required_scope: AuthScope) -> bool:
        """Check if token has required scope.

        Args:
            token_claims: Validated token claims
            required_scope: Required scope

        Returns:
            True if token has the scope
        """
        user_scopes = self.get_user_scopes(token_claims)
        return required_scope in user_scopes

    def has_any_scope(self, token_claims: Dict[str, Any], required_scopes: Set[AuthScope]) -> bool:
        """Check if token has any of the required scopes.

        Args:
            token_claims: Validated token claims
            required_scopes: Required scopes

        Returns:
            True if token has at least one required scope
        """
        user_scopes = self.get_user_scopes(token_claims)
        return bool(user_scopes & required_scopes)

    def has_all_scopes(self, token_claims: Dict[str, Any], required_scopes: Set[AuthScope]) -> bool:
        """Check if token has all required scopes.

        Args:
            token_claims: Validated token claims
            required_scopes: Required scopes

        Returns:
            True if token has all required scopes
        """
        user_scopes = self.get_user_scopes(token_claims)
        return required_scopes.issubset(user_scopes)


# Global service instance
auth_service = JWTAuthService()


def get_auth_service() -> JWTAuthService:
    """Dependency injection for auth service"""
    return auth_service
