"""
Service Level Objectives (SLOs) and Service Level Indicators (SLIs) for Goblin Assistant.

Defines critical user journeys and their performance targets for reliable operation.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from opentelemetry import metrics


@dataclass
class SLODefinition:
    """Definition of a Service Level Objective."""

    name: str
    description: str
    sli_name: str
    target_percentage: float  # e.g., 99.9 for 99.9%
    window_days: int = 28  # Rolling window for SLO calculation
    metadata: Dict[str, Any] = None


@dataclass
class SLIDefinition:
    """Definition of a Service Level Indicator."""

    name: str
    description: str
    metric_name: str
    good_events_query: str  # Query/filter for good events
    total_events_query: str  # Query/filter for total events
    unit: str = "1"  # Unit for the SLI (typically "1" for ratio)


# Critical SLOs for Goblin Assistant
SLO_DEFINITIONS = [
    SLODefinition(
        name="chat_response_time",
        description="Chat completion responses must be fast",
        sli_name="chat_response_time_sli",
        target_percentage=99.0,  # 99% of requests < 2s
        window_days=28,
        metadata={
            "user_journey": "chat_completion",
            "threshold_ms": 2000,
            "percentile": "p95",
        },
    ),
    SLODefinition(
        name="auth_success_rate",
        description="Authentication must be highly reliable",
        sli_name="auth_success_rate_sli",
        target_percentage=99.9,  # 99.9% success rate
        window_days=28,
        metadata={"user_journey": "authentication", "error_budget_percent": 0.1},
    ),
    SLODefinition(
        name="llm_provider_availability",
        description="LLM providers must be highly available",
        sli_name="llm_provider_availability_sli",
        target_percentage=99.5,  # 99.5% availability
        window_days=28,
        metadata={"user_journey": "llm_inference", "error_budget_percent": 0.5},
    ),
    SLODefinition(
        name="api_availability",
        description="Core API endpoints must be highly available",
        sli_name="api_availability_sli",
        target_percentage=99.9,  # 99.9% availability
        window_days=28,
        metadata={"user_journey": "api_access", "error_budget_percent": 0.1},
    ),
]

# SLI definitions with metric queries
SLI_DEFINITIONS = [
    SLIDefinition(
        name="chat_response_time_sli",
        description="Ratio of chat responses under 2 seconds",
        metric_name="http_request_duration_seconds",
        good_events_query='method="POST" AND path=~"/chat.*" AND le="2.0"',
        total_events_query='method="POST" AND path=~"/chat.*"',
        unit="1",
    ),
    SLIDefinition(
        name="auth_success_rate_sli",
        description="Ratio of successful authentication requests",
        metric_name="http_requests_total",
        good_events_query='method=~"POST|PUT" AND path=~"/auth.*" AND status_code=~"2.."',
        total_events_query='method=~"POST|PUT" AND path=~"/auth.*"',
        unit="1",
    ),
    SLIDefinition(
        name="llm_provider_availability_sli",
        description="Ratio of successful LLM provider calls",
        metric_name="llm_provider_requests_total",
        good_events_query='status="success"',
        total_events_query='status=~"success|error"',
        unit="1",
    ),
    SLIDefinition(
        name="api_availability_sli",
        description="Ratio of successful API requests",
        metric_name="http_requests_total",
        good_events_query='status_code=~"2.."',
        total_events_query='status_code=~"2..|4..|5.."',
        unit="1",
    ),
]


class SLOMonitor:
    """Monitor and track SLO compliance."""

    def __init__(self):
        self.meter = metrics.get_meter("goblin-assistant-slo")
        self.slo_gauges = {}
        self.sli_counters = {}

        # Initialize SLO gauges
        for slo in SLO_DEFINITIONS:
            self.slo_gauges[slo.name] = self.meter.create_gauge(
                name=f"slo_{slo.name}_compliance_ratio",
                description=f"SLO compliance ratio for {slo.name}",
                unit="1",
            )

        # Initialize SLI counters
        for sli in SLI_DEFINITIONS:
            self.sli_counters[f"{sli.name}_good"] = self.meter.create_counter(
                name=f"sli_{sli.name}_good_events_total",
                description=f"Good events for SLI {sli.name}",
                unit=sli.unit,
            )
            self.sli_counters[f"{sli.name}_total"] = self.meter.create_counter(
                name=f"sli_{sli.name}_total_events_total",
                description=f"Total events for SLI {sli.name}",
                unit=sli.unit,
            )

    def record_sli_event(self, sli_name: str, is_good: bool = True):
        """Record an SLI event (good or total)."""
        if is_good:
            counter = self.sli_counters.get(f"{sli_name}_good")
            if counter:
                counter.add(1)
        else:
            # For bad events, we increment total but not good
            counter = self.sli_counters.get(f"{sli_name}_total")
            if counter:
                counter.add(1)

        # Always increment total for ratio calculation
        total_counter = self.sli_counters.get(f"{sli_name}_total")
        if total_counter:
            total_counter.add(1)

    def update_slo_compliance(self, slo_name: str, compliance_ratio: float):
        """Update SLO compliance ratio (0.0 to 1.0)."""
        gauge = self.slo_gauges.get(slo_name)
        if gauge:
            gauge.set(compliance_ratio)


# Global SLO monitor instance
slo_monitor = SLOMonitor()


def get_slo_definitions() -> List[SLODefinition]:
    """Get all SLO definitions."""
    return SLO_DEFINITIONS.copy()


def get_sli_definitions() -> List[SLIDefinition]:
    """Get all SLI definitions."""
    return SLI_DEFINITIONS.copy()


def record_chat_response_time(duration_ms: float):
    """Record a chat response time event for SLO tracking."""
    is_good = duration_ms <= 2000  # 2 second threshold
    slo_monitor.record_sli_event("chat_response_time_sli", is_good)


def record_auth_attempt(success: bool):
    """Record an authentication attempt for SLO tracking."""
    slo_monitor.record_sli_event("auth_success_rate_sli", success)


def record_llm_provider_call(success: bool):
    """Record an LLM provider call for SLO tracking."""
    slo_monitor.record_sli_event("llm_provider_availability_sli", success)


def record_api_request(status_code: int):
    """Record an API request for SLO tracking."""
    is_success = 200 <= status_code < 400
    slo_monitor.record_sli_event("api_availability_sli", is_success)
