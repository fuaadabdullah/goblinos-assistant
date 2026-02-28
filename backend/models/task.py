"""
Database model for tasks.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    Integer,
    DateTime,
)
from database import Base


class Task(Base):
    __tablename__ = "tasks"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=True)
    goblin = Column(String(50), nullable=False)
    task = Column(Text, nullable=False)
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    result = Column(Text)
    cost = Column(Float, default=0.0)
    tokens = Column(Integer, default=0)
    duration_ms = Column(Integer, default=0)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "goblin": self.goblin,
            "task": self.task,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "result": self.result,
            "cost": self.cost,
            "tokens": self.tokens,
            "duration_ms": self.duration_ms,
        }
