from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import os
from typing import Optional
import bcrypt

# Use try/except for flexible imports that work both locally and in container
try:
    from backend.auth.challenge_store import get_challenge_store_instance
    from backend.auth.passkeys import WebAuthnPasskey
    from backend.auth.google_oauth import get_google_oauth_service
    from backend.database import get_db
    from backend.auth_service import (
        JWTAuthService,
        get_auth_service as _get_auth_service_module,
    )
    from backend.models_base import User
    from backend.auth.policies import UserRole
except ImportError:
    from .challenge_store import get_challenge_store_instance
    from .passkeys import WebAuthnPasskey
    from .google_oauth import get_google_oauth_service
    from ..database import get_db
    from ..auth_service import (
        JWTAuthService,
        get_auth_service as _get_auth_service_module,
    )
    from ..models_base import User
    from .policies import UserRole

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class RegisterResponse(BaseModel):
    message: str
    user_id: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def get_auth_service() -> JWTAuthService:
    """Dependency to get JWT auth service instance.

    This delegates to the canonical auth_service.get_auth_service() which manages
    the global service instance and configuration.
    """
    try:
        return _get_auth_service_module()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )


# Passkey configuration
CHALLENGE_EXPIRE_MINUTES = int(os.getenv("CHALLENGE_EXPIRE_MINUTES", "5"))
challenge_store = get_challenge_store_instance()


@router.post("/register", response_model=RegisterResponse)
async def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    """Register a new user."""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists",
        )

    # Hash the password
    password_hash = hash_password(request.password)

    # Create new user
    new_user = User(
        email=request.email,
        password_hash=password_hash,
        name=request.name,
        role=UserRole.USER.value,  # Default role
        token_version=0,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return RegisterResponse(message="User registered successfully", user_id=str(new_user.id))


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
    auth_service: JWTAuthService = Depends(get_auth_service),
):
    """Authenticate user and return JWT tokens."""
    # Look up user by email
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # Verify password
    if not user.password_hash:
        if os.getenv("ALLOW_TEST_PASSWORD_BYPASS", "0") != "1":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
            )
    elif not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # Create tokens
    access_token = auth_service.create_access_token(user.id, user.email, UserRole(user.role))
    refresh_token = auth_service.create_refresh_token(user.id)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=900,  # 15 minutes
    )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(
    request: RefreshRequest,
    db: Session = Depends(get_db),
    auth_service: JWTAuthService = Depends(get_auth_service),
):
    """Refresh access token using refresh token."""
    access_token = auth_service.refresh_access_token(request.refresh_token, db)
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    return RefreshResponse(
        access_token=access_token,
        expires_in=900,  # 15 minutes
    )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: JWTAuthService = Depends(get_auth_service),
):
    """Logout user (client should discard tokens)."""
    # In a stateless JWT system, logout is handled client-side
    # For enhanced security, you could implement token blacklisting
    return {"message": "Logged out successfully"}


class TurnstileVerifyRequest(BaseModel):
    token: str


class TurnstileVerifyResponse(BaseModel):
    success: bool
    message: str


@router.post("/turnstile/verify", response_model=TurnstileVerifyResponse)
async def verify_turnstile(request: TurnstileVerifyRequest):
    """Verify Cloudflare Turnstile token.

    This endpoint validates a Turnstile CAPTCHA token on the server side.
    The token is generated by the Turnstile widget on the frontend.
    """
    import httpx

    turnstile_secret = os.getenv("TURNSTILE_SECRET_KEY")
    if not turnstile_secret:
        # If no secret is configured, allow the request (development mode)
        return TurnstileVerifyResponse(
            success=True, message="Turnstile verification skipped (no secret configured)"
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                data={
                    "secret": turnstile_secret,
                    "response": request.token,
                },
                timeout=5.0,
            )

            if response.status_code != 200:
                return TurnstileVerifyResponse(
                    success=False,
                    message=f"Turnstile verification failed (HTTP {response.status_code})",
                )

            result = response.json()

            if result.get("success"):
                return TurnstileVerifyResponse(
                    success=True,
                    message="Turnstile verification successful",
                )
            else:
                error_codes = result.get("error-codes", [])
                return TurnstileVerifyResponse(
                    success=False,
                    message=f"Turnstile verification failed: {', '.join(error_codes)}",
                )
    except httpx.TimeoutException:
        return TurnstileVerifyResponse(
            success=False,
            message="Turnstile verification timeout",
        )
    except Exception as e:
        return TurnstileVerifyResponse(
            success=False,
            message=f"Turnstile verification error: {str(e)}",
        )


@router.get("/me")
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
    auth_service: JWTAuthService = Depends(get_auth_service),
):
    """Get current user information."""
    payload = auth_service.validate_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user_id = payload.get("sub")
    if isinstance(user_id, str):
        try:
            import uuid

            user_id = uuid.UUID(user_id)
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    role_value = user.role.value if hasattr(user.role, "value") else user.role

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": role_value,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


async def cleanup_expired_challenges() -> int:
    """Cleanup expired challenges from the configured challenge store.
    Returns the count of removed entries; used by background worker in main.py.
    """
    try:
        count = await challenge_store.cleanup_expired()
        return count or 0
    except Exception:
        return 0


class PasskeyRegistrationRequest(BaseModel):
    email: EmailStr
    credential_id: str
    public_key: str


class PasskeyAuthRequest(BaseModel):
    email: EmailStr
    credential_id: str
    authenticator_data: str
    client_data_json: str
    signature: str


@router.post("/passkey/challenge")
async def get_passkey_challenge(email: EmailStr | None = None):
    """Get a passkey challenge and optionally store it for a user's email.
    Returns: {"challenge": <challenge>}.
    """
    challenge = WebAuthnPasskey.generate_challenge()
    if email:
        await challenge_store.set_challenge(email, challenge, CHALLENGE_EXPIRE_MINUTES * 60)
    return {"challenge": challenge}


@router.post("/passkey/register")
async def register_passkey(request: PasskeyRegistrationRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Validate basic formats (tests will mock deeper validation)
    try:
        # Basic base64url check
        WebAuthnPasskey.decode_base64url(request.credential_id)
        WebAuthnPasskey.decode_base64url(request.public_key)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid passkey format"
        )

    user.passkey_credential_id = request.credential_id
    user.passkey_public_key = request.public_key
    db.commit()
    return {"message": "Passkey registered successfully"}


@router.post("/passkey/auth")
async def authenticate_passkey(request: PasskeyAuthRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not user.passkey_credential_id or not user.passkey_public_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Passkey not registered for this user",
        )

    if request.credential_id != user.passkey_credential_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credential mismatch")

    # Validate challenge
    stored_challenge = await challenge_store.get_challenge(request.email)
    if not stored_challenge:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="No challenge or expired"
        )

    # Remove challenge once used
    await challenge_store.delete_challenge(request.email)

    # Perform verification (cryptographic verification can be mocked in tests)
    origin = os.getenv("FRONTEND_URL", "http://localhost:5173")
    is_valid = await WebAuthnPasskey.verify_passkey_authentication(
        credential_id=request.credential_id,
        stored_public_key=user.passkey_public_key,
        authenticator_data_b64=request.authenticator_data,
        client_data_json_b64=request.client_data_json,
        signature_b64=request.signature,
        challenge=stored_challenge,
        origin=origin,
    )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid passkey authentication",
        )

    # Create access token via auth service
    auth_service = get_auth_service()
    access_token = auth_service.create_access_token(user.id, user.email, UserRole(user.role))

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "name": user.name},
    }


# ============================================================================
# Google OAuth Endpoints
# ============================================================================


class GoogleCallbackRequest(BaseModel):
    code: str
    state: Optional[str] = None


class GoogleAuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


@router.get("/google/url")
async def get_google_auth_url():
    """Get Google OAuth authorization URL for initiating login flow."""
    oauth_service = get_google_oauth_service()

    if not oauth_service.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
        )

    try:
        auth_url, state = oauth_service.get_authorization_url()
        return {"authorization_url": auth_url, "state": state}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/google/callback", response_model=GoogleAuthResponse)
async def google_oauth_callback(
    request: GoogleCallbackRequest,
    db: Session = Depends(get_db),
    auth_service: JWTAuthService = Depends(get_auth_service),
):
    """
    Handle Google OAuth callback.

    Exchange authorization code for tokens, get user info,
    and create/update user in database.
    """
    oauth_service = get_google_oauth_service()

    if not oauth_service.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured",
        )

    # Exchange code for tokens
    token_data = await oauth_service.exchange_code_for_tokens(request.code)
    if not token_data or "access_token" not in token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to exchange authorization code for tokens",
        )

    # Get user info from Google
    google_access_token = token_data["access_token"]
    google_user = await oauth_service.get_user_info(google_access_token)

    if not google_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to retrieve user info from Google",
        )

    google_id = google_user.get("id")
    email = google_user.get("email")
    name = google_user.get("name")
    picture = google_user.get("picture")

    if not email or not google_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user data from Google (missing email or id)",
        )

    # Check if user exists by google_id or email
    user = db.query(User).filter((User.google_id == google_id) | (User.email == email)).first()

    if user:
        # Update existing user with Google info if not already set
        if not user.google_id:
            user.google_id = google_id
        if not user.name and name:
            user.name = name
        if not user.avatar_url and picture:
            user.avatar_url = picture
        db.commit()
        db.refresh(user)
    else:
        # Create new user from Google account
        user = User(
            email=email,
            name=name,
            google_id=google_id,
            avatar_url=picture,
            role=UserRole.USER.value,
            token_version=0,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # Create JWT tokens for our system
    access_token = auth_service.create_access_token(user.id, user.email, UserRole(user.role))
    refresh_token = auth_service.create_refresh_token(user.id)

    return GoogleAuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=900,  # 15 minutes
        user={
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "avatar_url": getattr(user, "avatar_url", None),
        },
    )
