"""Small compatibility shim for router imports.

Routers can import the factories they need from here without deep try/except
ladders or sys.path mutation. All functions are thin wrappers around the
canonical implementations.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import Depends
from sqlalchemy.orm import Session

from backend.auth_service import get_auth_service
from backend.config import settings
from backend.database import get_db
from backend.gateway_service import get_gateway_service
from backend.services.gateway_exceptions import TokenBudgetExceeded, MaxTokensExceeded
from backend.services.routing import RoutingService
from backend.services.routing_compat import (
    get_routing_service_compat as get_routing_service,
)

# Keep encryption key behavior aligned with routing_router.py
_ROUTING_ENCRYPTION_KEY = os.getenv("ROUTING_ENCRYPTION_KEY", "default-dev-key-change-me")


def get_chat_routing_service(db: Session = Depends(get_db)) -> RoutingService:
    """Chat endpoint routing dependency.

    Chat completion flow expects a routing service that returns a provider + model
    decision. Use the legacy (DB-backed) RoutingService, not the unified compat layer.
    """
    return RoutingService(db, _ROUTING_ENCRYPTION_KEY)

__all__ = [
    "get_auth_service",
    "settings",
    "get_gateway_service",
    "TokenBudgetExceeded",
    "MaxTokensExceeded",
    "get_routing_service",
    "get_chat_routing_service",
]
