from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    JSON,
    Float,
    ForeignKey,
    UUID,
)
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

# Import Base from database.py to use the same declarative base
try:
    # Package import (preferred when running as `backend.*`)
    from .database import Base
except ImportError:  # pragma: no cover - support flat imports in tests/scripts
    # Flat import (used when `/app/backend` is on sys.path)
    from database import Base  # type: ignore


class User(Base):
    __tablename__ = "app_users"  # Updated to match consolidated auth schema

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String)  # Kept for backward compatibility during migration
    name = Column(String)
    avatar_url = Column(String)  # Profile picture URL (e.g. from Google OAuth)
    google_id = Column(String, unique=True)  # Kept for backward compatibility
    passkey_credential_id = Column(String)  # Kept for backward compatibility
    passkey_public_key = Column(Text)  # Kept for backward compatibility

    # New consolidated auth fields
    role = Column(String, default="user")  # Simple role for RBAC
    token_version = Column(Integer, default=0)  # For token revocation

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    tasks = relationship("Task", back_populates="user")
    streams = relationship(
        "Stream", back_populates="streams_user"
    )  # Updated to avoid conflict
    sessions = relationship("UserSession", back_populates="user")
    role_assignments = relationship(
        "UserRoleAssignment",
        back_populates="user",
        foreign_keys="UserRoleAssignment.user_id",
    )


class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("app_users.id"), nullable=False)
    refresh_token_id = Column(String, unique=True, nullable=False)
    device_info = Column(Text)
    ip_address = Column(String)
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    revoked = Column(Boolean, default=False)
    revoked_at = Column(DateTime)
    revoked_reason = Column(Text)

    # Relationships
    user = relationship("User", back_populates="sessions")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    actor_id = Column(UUID(as_uuid=True), ForeignKey("app_users.id"))
    action = Column(String, nullable=False)
    object_table = Column(String)
    object_id = Column(String)
    old_values = Column(JSON)
    new_values = Column(JSON)
    metadata_ = Column(JSON)  # Renamed to avoid conflict with SQLAlchemy metadata
    ip_address = Column(String)
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserRole(Base):
    __tablename__ = "user_roles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    permissions = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    assignments = relationship("UserRoleAssignment", back_populates="role")


class UserRoleAssignment(Base):
    __tablename__ = "user_role_assignments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("app_users.id"), nullable=False)
    role_id = Column(UUID(as_uuid=True), ForeignKey("user_roles.id"), nullable=False)
    assigned_by = Column(UUID(as_uuid=True), ForeignKey("app_users.id"))
    assigned_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

    # Relationships
    user = relationship(
        "User", back_populates="role_assignments", foreign_keys=[user_id]
    )
    role = relationship("UserRole", back_populates="assignments")
    assigner = relationship("User", foreign_keys=[assigned_by])


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("app_users.id")
    )  # Updated to match new User model
    goblin = Column(String, nullable=False)
    task = Column(Text, nullable=False)
    code = Column(Text)
    provider = Column(String)
    model = Column(String)
    status = Column(
        String, default="queued"
    )  # queued, running, completed, failed, cancelled
    result = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="tasks")


class Stream(Base):
    __tablename__ = "streams"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("app_users.id")
    )  # Updated to match new User model
    goblin = Column(String, nullable=False)
    task = Column(Text, nullable=False)
    code = Column(Text)
    provider = Column(String)
    model = Column(String)
    status = Column(String, default="running")  # running, completed, cancelled
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    streams_user = relationship(
        "User", back_populates="streams"
    )  # Renamed to avoid conflict
    chunks = relationship(
        "StreamChunk", back_populates="stream", cascade="all, delete-orphan"
    )


class StreamChunk(Base):
    __tablename__ = "stream_chunks"

    id = Column(Integer, primary_key=True, index=True)
    stream_id = Column(String, ForeignKey("streams.id"), nullable=False)
    content = Column(Text)
    token_count = Column(Integer)
    cost_delta = Column(Float)
    done = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    stream = relationship("Stream", back_populates="chunks")


class SearchCollection(Base):
    __tablename__ = "search_collections"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    documents = relationship(
        "SearchDocument", back_populates="collection", cascade="all, delete-orphan"
    )


class SearchDocument(Base):
    __tablename__ = "search_documents"

    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("search_collections.id"), nullable=False)
    document_id = Column(String, nullable=False)  # External document ID
    document = Column(Text, nullable=False)
    document_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    collection = relationship("SearchCollection", back_populates="documents")

    __table_args__ = (
        {"schema": None},  # Default schema
    )


class SupportMessage(Base):
    __tablename__ = "support_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("app_users.id"), nullable=True)
    message = Column(Text, nullable=False)
    status = Column(String, default="open")  # open | in_progress | resolved | closed
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User")
