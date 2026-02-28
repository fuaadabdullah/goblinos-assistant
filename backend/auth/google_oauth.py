"""
Google OAuth Service
Handles Google OAuth authentication flow
"""

import os
import secrets
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
import structlog

logger = structlog.get_logger()


class GoogleOAuthService:
    """
    Google OAuth 2.0 authentication service.

    Handles authorization URL generation, token exchange,
    and user info retrieval from Google APIs.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ):
        self.client_id = client_id or os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("GOOGLE_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv(
            "GOOGLE_REDIRECT_URI",
            os.getenv("FRONTEND_URL", "http://localhost:3000")
            + "/auth/google/callback",
        )

        # Google OAuth endpoints
        self.auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        self.userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        self.tokeninfo_url = "https://oauth2.googleapis.com/tokeninfo"

    @property
    def is_configured(self) -> bool:
        """Check if Google OAuth is properly configured."""
        return bool(self.client_id and self.client_secret)

    def get_authorization_url(
        self,
        state: Optional[str] = None,
        scopes: Optional[list[str]] = None,
    ) -> tuple[str, str]:
        """
        Generate Google OAuth authorization URL.

        Args:
            state: Optional CSRF state token (auto-generated if None)
            scopes: OAuth scopes (defaults to openid, email, profile)

        Returns:
            Tuple of (authorization_url, state_token)
        """
        if not self.client_id:
            raise ValueError("GOOGLE_CLIENT_ID not configured")

        state_token = state or secrets.token_urlsafe(32)
        default_scopes = ["openid", "email", "profile"]

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(scopes or default_scopes),
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent",
            "state": state_token,
        }

        url = f"{self.auth_url}?{urlencode(params)}"
        return url, state_token

    async def exchange_code_for_tokens(
        self,
        code: str,
    ) -> Optional[dict[str, Any]]:
        """
        Exchange authorization code for access and refresh tokens.

        Args:
            code: Authorization code from OAuth callback

        Returns:
            Token response dict or None on failure
        """
        if not self.client_id or not self.client_secret:
            logger.error("Google OAuth not configured")
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.token_url,
                    data={
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "code": code,
                        "grant_type": "authorization_code",
                        "redirect_uri": self.redirect_uri,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                if response.status_code == 200:
                    return response.json()

                logger.error(
                    "Google token exchange failed",
                    status=response.status_code,
                    error=response.text,
                )
                return None

        except Exception as e:
            logger.exception("Exception during Google token exchange", error=str(e))
            return None

    async def get_user_info(
        self,
        access_token: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get user info from Google using access token.

        Args:
            access_token: Valid Google access token

        Returns:
            User info dict with id, email, name, picture, etc.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.userinfo_url,
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                if response.status_code == 200:
                    return response.json()

                logger.warning(
                    "Failed to get Google user info",
                    status=response.status_code,
                )
                return None

        except Exception as e:
            logger.exception("Exception getting Google user info", error=str(e))
            return None

    async def verify_access_token(
        self,
        access_token: str,
    ) -> Optional[dict[str, Any]]:
        """
        Verify a Google access token and get token info.

        Args:
            access_token: Access token to verify

        Returns:
            Token info dict or None if invalid
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.tokeninfo_url,
                    params={"access_token": access_token},
                )

                if response.status_code == 200:
                    return response.json()

                return None

        except Exception:
            return None


# Singleton instance
_google_oauth_service: Optional[GoogleOAuthService] = None


def get_google_oauth_service() -> GoogleOAuthService:
    """Get or create Google OAuth service singleton."""
    global _google_oauth_service
    if _google_oauth_service is None:
        _google_oauth_service = GoogleOAuthService()
    return _google_oauth_service
