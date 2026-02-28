from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import get_db
from .models import SupportMessage, User
from .auth_service import get_auth_service, JWTAuthService

router = APIRouter(prefix="/support", tags=["support"])
security = HTTPBearer(auto_error=False)


class SupportMessageRequest(BaseModel):
    message: str


class SupportMessageResponse(BaseModel):
    message_id: str
    status: str


def resolve_user_id(
    credentials: HTTPAuthorizationCredentials | None,
    db: Session,
    auth_service: JWTAuthService,
) -> str | None:
    if not credentials:
        return None

    claims = auth_service.validate_access_token(credentials.credentials)
    if not claims:
        return None

    user_id = claims.get("sub")
    if not user_id:
        return None

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return None

    return str(user.id)


@router.post("/message", response_model=SupportMessageResponse)
async def send_support_message(
    request: SupportMessageRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    auth_service: JWTAuthService = Depends(get_auth_service),
):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    user_id = resolve_user_id(credentials, db, auth_service)

    support_message = SupportMessage(
        user_id=user_id,
        message=message,
        status="open",
        user_agent=http_request.headers.get("user-agent"),
        ip_address=http_request.client.host if http_request.client else None,
    )

    db.add(support_message)
    db.commit()
    db.refresh(support_message)

    return SupportMessageResponse(message_id=str(support_message.id), status="received")
