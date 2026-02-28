"""
Database models for providers, metrics, and policies.
"""

import sys

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    JSON,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base  # Import Base from the correct location

# Ensure a single module instance across import paths.
if __name__ == "models.provider":
    sys.modules.setdefault("backend.models.provider", sys.modules[__name__])
elif __name__ == "backend.models.provider":
    sys.modules.setdefault("models.provider", sys.modules[__name__])


class Provider(Base):
    """Represents a provider (OpenAI, Anthropic, etc.) with capabilities and configuration."""

    __tablename__ = "providers"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    name = Column(
        String(100), nullable=False, unique=True
    )  # e.g., "openai", "anthropic"
    display_name = Column(String(200), nullable=True)  # e.g., "OpenAI", "Anthropic"
    api_key = Column(String(500), nullable=True)  # From api/app.py
    base_url = Column(String(500), nullable=True)  # From api/app.py
    api_key_encrypted = Column(Text, nullable=True)  # Encrypted API key
    is_active = Column(Boolean, default=True)
    enabled = Column(Boolean, default=True)  # From api/app.py
    capabilities = Column(JSON, nullable=True)  # List of supported capabilities
    models = Column(JSON, nullable=True)  # List of available models, from api/app.py
    rate_limits = Column(JSON, nullable=True)  # Rate limiting configuration
    cost_per_token = Column(Float, nullable=True)  # Cost per token in USD
    priority = Column(Integer, default=0)  # Higher priority = preferred
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    metrics = relationship(
        "ProviderMetric", back_populates="provider", cascade="all, delete-orphan"
    )
    policies = relationship(
        "ProviderPolicy", back_populates="provider", cascade="all, delete-orphan"
    )
    credentials = relationship(
        "ProviderCredential", back_populates="provider", cascade="all, delete-orphan"
    )
    model_configs = relationship(
        "ModelConfig", back_populates="provider", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "name": self.name,
            "api_key": self.api_key or "",
            "base_url": self.base_url or "",
            "models": self.models or [],
            "enabled": self.enabled,
        }

    def __repr__(self):
        return f"<Provider(name='{self.name}', active={self.is_active})>"


class ProviderMetric(Base):
    """Metrics collected for provider health monitoring."""

    __tablename__ = "provider_metrics"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(Integer, ForeignKey("providers.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Health metrics
    is_healthy = Column(Boolean, nullable=False)
    response_time_ms = Column(Float, nullable=True)  # Average response time
    error_rate = Column(Float, default=0.0)  # Error rate (0.0 to 1.0)
    throughput_rpm = Column(Float, nullable=True)  # Requests per minute

    # Resource metrics
    tokens_used = Column(Integer, nullable=True)
    cost_incurred = Column(Float, nullable=True)

    # Additional metadata
    metadata_json = Column(JSON, nullable=True)  # Additional metric data

    # Relationships
    provider = relationship("Provider", back_populates="metrics")


class ProviderPolicy(Base):
    """Policies for routing decisions (fallback, load balancing, etc.)."""

    __tablename__ = "provider_policies"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(Integer, ForeignKey("providers.id"), nullable=False)
    name = Column(String(100), nullable=False)  # e.g., "fallback", "load_balance"
    policy_type = Column(
        String(50), nullable=False
    )  # "fallback", "load_balance", "circuit_breaker"
    conditions = Column(JSON, nullable=False)  # Conditions for policy activation
    actions = Column(JSON, nullable=False)  # Actions to take when conditions met
    priority = Column(Integer, default=0)  # Policy priority
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    provider = relationship("Provider", back_populates="policies")

    def __repr__(self):
        return f"<ProviderPolicy(provider_id={self.provider_id}, name='{self.name}', type='{self.policy_type}')>"


class ProviderCredential(Base):
    __tablename__ = "provider_credentials"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(Integer, ForeignKey("providers.id"))
    encrypted_key = Column(Text)  # Encrypted API key using cryptography
    scopes = Column(JSON, nullable=True)  # e.g., ["read", "write"]
    created_by = Column(String)  # User ID or "system"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)

    provider = relationship("Provider", back_populates="credentials")


class ModelConfig(Base):
    __tablename__ = "model_configs"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(Integer, ForeignKey("providers.id"))
    name = Column(String)  # e.g., "gpt-4", "claude-3"
    params = Column(JSON)  # Model parameters like max_tokens, temperature
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    provider = relationship("Provider", back_populates="model_configs")


class RoutingRequest(Base):
    """Log of routing requests for analytics and debugging."""

    __tablename__ = "routing_requests"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), nullable=False, unique=True)
    capability = Column(String(100), nullable=False)
    requirements = Column(JSON, nullable=True)  # Specific requirements for the request
    selected_provider_id = Column(Integer, ForeignKey("providers.id"), nullable=True)
    response_time_ms = Column(Float, nullable=True)
    success = Column(Boolean, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    selected_provider = relationship("Provider")

    def __repr__(self):
        return f"<RoutingRequest(id='{self.request_id}', capability='{self.capability}', success={self.success})>"
