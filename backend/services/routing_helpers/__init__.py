"""
Routing helpers module - backward compatibility layer.

This module re-exports all functions from sub-modules to maintain
backward compatibility with existing imports.
"""

# Autoscaling and rate limiting
from .autoscaling import (
    handle_autoscaling_and_emergency_routing,
    check_autoscaling_conditions,
)

# Health and metrics
from .health_metrics import (
    fetch_recent_provider_metrics,
    fetch_recent_provider_metrics_limited,
    calculate_health_rate,
    calculate_average_response_time,
    calculate_response_time_score,
    calculate_overall_health_score,
    calculate_provider_health_score,
    calculate_performance_bonus_from_response_time,
    calculate_provider_performance_bonus,
)

# SLA compliance
from .sla_compliance import (
    check_provider_sla_compliance,
    calculate_sla_compliance_rate,
    should_use_fallback_based_on_conditions,
    should_use_fallback_based_on_compliance,
    should_use_latency_fallback,
    check_sla_compliance_with_monitor,
    calculate_sla_score_from_compliance,
    calculate_provider_sla_score,
)

# Cost analysis
from .cost_analysis import (
    calculate_cost_penalty,
    calculate_cost_penalty_with_budget,
    calculate_capability_bonus,
)

# Requirements checking
from .requirements import (
    check_model_requirement,
    check_context_window_requirement,
    check_vision_capability_requirement,
    check_provider_requirements,
    extract_local_llm_routing_parameters,
)

# Local LLM routing
from .local_llm import (
    find_ollama_provider,
    build_local_routing_result,
    handle_local_llm_routing,
    process_local_llm_routing_result,
    find_fast_local_model,
    get_fallback_provider_info,
)

# Provider selection
from .provider_selection import (
    handle_provider_selection_and_fallback,
    handle_routing_error,
)

__all__ = [
    # Autoscaling
    "handle_autoscaling_and_emergency_routing",
    "check_autoscaling_conditions",
    # Health metrics
    "fetch_recent_provider_metrics",
    "fetch_recent_provider_metrics_limited",
    "calculate_health_rate",
    "calculate_average_response_time",
    "calculate_response_time_score",
    "calculate_overall_health_score",
    "calculate_provider_health_score",
    "calculate_performance_bonus_from_response_time",
    "calculate_provider_performance_bonus",
    # SLA compliance
    "check_provider_sla_compliance",
    "calculate_sla_compliance_rate",
    "should_use_fallback_based_on_conditions",
    "should_use_fallback_based_on_compliance",
    "should_use_latency_fallback",
    "check_sla_compliance_with_monitor",
    "calculate_sla_score_from_compliance",
    "calculate_provider_sla_score",
    # Cost analysis
    "calculate_cost_penalty",
    "calculate_cost_penalty_with_budget",
    "calculate_capability_bonus",
    # Requirements
    "check_model_requirement",
    "check_context_window_requirement",
    "check_vision_capability_requirement",
    "check_provider_requirements",
    "extract_local_llm_routing_parameters",
    # Local LLM
    "find_ollama_provider",
    "build_local_routing_result",
    "handle_local_llm_routing",
    "process_local_llm_routing_result",
    "find_fast_local_model",
    "get_fallback_provider_info",
    # Provider selection
    "handle_provider_selection_and_fallback",
    "handle_routing_error",
]
