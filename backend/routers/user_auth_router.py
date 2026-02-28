# apps/goblin-assistant-root/backend/routers/user_auth_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from ..database import get_db
from models import User  # Assuming User model is now in backend/models/user.py


router = APIRouter()

# Pydantic models for request/response
class UserRegistration(BaseModel):
    email: EmailStr
    password: str # In a real app, this would be hashed
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str # In a real app, this would be hashed

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    name: str

class AuthSuccessResponse(BaseModel):
    token: str
    user: UserResponse

@router.post("/auth/register", response_model=AuthSuccessResponse, tags=["auth"])
async def register(user_data: UserRegistration, db: Session = Depends(get_db)):
    """Register a new user"""
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already exists"
        )

    new_user = User(
        email=user_data.email,
        name=user_data.name or user_data.email.split("@")[0].title()
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return AuthSuccessResponse(
        token=f"jwt_token_{new_user.id}",
        user=UserResponse(id=new_user.id, email=new_user.email, name=new_user.name)
    )

@router.post("/auth/login", response_model=AuthSuccessResponse, tags=["auth"])
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login a user"""
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # In a real app, verify hashed password
    # For now, just check if password is provided
    if not user_data.password: # Mock check, replace with actual password verification
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )


    user.last_login = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    return AuthSuccessResponse(
        token=f"jwt_token_{user.id}",
        user=UserResponse(id=user.id, email=user.email, name=user.name)
    )

@router.post("/auth/logout", tags=["auth"])
async def logout():
    """Logout a user"""
    return {"message": "Logged out successfully"}

@router.post("/auth/validate", tags=["auth"])
async def validate():
    """Validate authentication token"""
    return {
        "valid": True,
        "user": {"id": "mock_user", "email": "user@example.com"},
    }
