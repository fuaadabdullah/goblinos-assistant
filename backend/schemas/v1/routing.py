from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RouteRequest(BaseModel):
    capability: str
    requirements: Optional[Dict[str, Any]] = None
    prefer_cost: Optional[bool] = False
    max_retries: Optional[int] = 2


class EnhancedRouteRequest(BaseModel):
    capability: str
    requirements: Optional[Dict[str, Any]] = None
    sla_target_ms: Optional[float] = None
    cost_budget: Optional[float] = None
    latency_priority: Optional[str] = None
    user_region: Optional[str] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    request_content: Optional[str] = None


class ProviderInfo(BaseModel):
    id: str
    name: str
    display_name: str
    capabilities: List[str]
    models: List[Dict[str, Any]]
    priority: int
    is_active: bool
    score: Optional[float] = None

