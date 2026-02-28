"""
Database model for models.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
)
from database import Base


class Model(Base):
    __tablename__ = "models"
    __table_args__ = {"extend_existing": True}
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    provider = Column(String(50), nullable=False)
    model_id = Column(String(100), nullable=False)
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=4096)
    enabled = Column(Boolean, default=True)

    def to_dict(self):
        return {
            "name": self.name,
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enabled": self.enabled,
        }
