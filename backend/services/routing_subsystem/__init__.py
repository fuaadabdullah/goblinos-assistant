"""
Routing subsystem for unified provider management and intelligent request routing.

This package consolidates routing, scaling, monitoring, and provider management
into a single cohesive subsystem.
"""

from .manager import RoutingManager, get_routing_manager, routing_lifecycle
from .decision_engine import DecisionEngine, get_decision_engine, RoutingDecision
from .scoring import ProviderScore, score_provider
from .provider_health import ProviderHealthMonitor, get_provider_health_monitor
from .cache import RoutingCache, get_routing_cache
from .policies import RoutingPolicy, PolicyManager, get_policy_manager

__all__ = [
    # Main interfaces
    "RoutingManager",
    "get_routing_manager",
    "routing_lifecycle",
    # Decision engine
    "DecisionEngine",
    "get_decision_engine",
    "RoutingDecision",
    # Scoring
    "ProviderScore",
    "score_provider",
    # Policies
    "PolicyManager",
    "get_policy_manager",
    "RoutingPolicy",
    # Cache
    "RoutingCache",
    "get_routing_cache",
    # Health monitoring
    "ProviderHealthMonitor",
    "get_provider_health_monitor",
]
