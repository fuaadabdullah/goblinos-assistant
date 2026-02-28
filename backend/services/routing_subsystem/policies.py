"""
Routing policies for configurable routing behavior.

Provides policy-driven routing configuration that can be changed
without code modifications.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy types."""

    LATENCY_FIRST = "latency_first"
    COST_FIRST = "cost_first"
    RELIABILITY_FIRST = "reliability_first"
    BALANCED = "balanced"
    CUSTOM = "custom"


@dataclass
class RoutingPolicy:
    """Configuration for routing behavior."""

    name: str
    strategy: RoutingStrategy
    weights: Dict[str, float]
    constraints: Dict[str, Any]
    fallbacks: list[str]
    enabled: bool = True

    def __post_init__(self):
        """Validate policy configuration."""
        required_weights = ["latency", "cost", "reliability", "capability"]
        for weight in required_weights:
            if weight not in self.weights:
                self.weights[weight] = 0.25  # Default equal weighting

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}


class PolicyManager:
    """Manager for routing policies."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize policy manager.

        Args:
            config_file: Path to policy configuration file
        """
        self.policies: Dict[str, RoutingPolicy] = {}
        self.config_file = config_file or os.path.join(
            os.path.dirname(__file__), "policies.json"
        )
        self._load_policies()

    def _load_policies(self):
        """Load policies from configuration file."""
        # Default policies
        self._load_default_policies()

        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)

                for policy_name, policy_config in config.items():
                    try:
                        strategy = RoutingStrategy(
                            policy_config.get("strategy", "balanced")
                        )
                        policy = RoutingPolicy(
                            name=policy_name,
                            strategy=strategy,
                            weights=policy_config.get("weights", {}),
                            constraints=policy_config.get("constraints", {}),
                            fallbacks=policy_config.get("fallbacks", []),
                            enabled=policy_config.get("enabled", True),
                        )
                        self.policies[policy_name] = policy
                        logger.info(f"Loaded policy: {policy_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load policy {policy_name}: {e}")

            except Exception as e:
                logger.warning(f"Failed to load policy config: {e}")

    def _load_default_policies(self):
        """Load default routing policies."""
        defaults = {
            "latency_first": RoutingPolicy(
                name="latency_first",
                strategy=RoutingStrategy.LATENCY_FIRST,
                weights={
                    "latency": 0.6,
                    "cost": 0.1,
                    "reliability": 0.2,
                    "capability": 0.1,
                },
                constraints={"max_latency_ms": 1000},
                fallbacks=["balanced", "cost_first"],
            ),
            "cost_first": RoutingPolicy(
                name="cost_first",
                strategy=RoutingStrategy.COST_FIRST,
                weights={
                    "latency": 0.1,
                    "cost": 0.6,
                    "reliability": 0.2,
                    "capability": 0.1,
                },
                constraints={"max_cost_per_request": 0.01},
                fallbacks=["balanced", "latency_first"],
            ),
            "reliability_first": RoutingPolicy(
                name="reliability_first",
                strategy=RoutingStrategy.RELIABILITY_FIRST,
                weights={
                    "latency": 0.2,
                    "cost": 0.1,
                    "reliability": 0.6,
                    "capability": 0.1,
                },
                constraints={"min_success_rate": 0.95},
                fallbacks=["balanced", "latency_first"],
            ),
            "balanced": RoutingPolicy(
                name="balanced",
                strategy=RoutingStrategy.BALANCED,
                weights={
                    "latency": 0.3,
                    "cost": 0.3,
                    "reliability": 0.3,
                    "capability": 0.1,
                },
                constraints={},
                fallbacks=["latency_first", "cost_first"],
            ),
        }

        self.policies.update(defaults)

    def get_policy(self, name: str) -> Optional[RoutingPolicy]:
        """Get a policy by name.

        Args:
            name: Policy name

        Returns:
            RoutingPolicy or None if not found
        """
        return self.policies.get(name)

    def get_active_policies(self) -> list[RoutingPolicy]:
        """Get all active (enabled) policies.

        Returns:
            List of active RoutingPolicy objects
        """
        return [p for p in self.policies.values() if p.enabled]

    def apply_policy(self, provider_scores: list, policy: dict) -> list:
        """Apply policy constraints and sorting to provider scores.

        Args:
            provider_scores: List of provider score dictionaries
            policy: Policy configuration dictionary

        Returns:
            Filtered and sorted list of provider scores
        """
        if not policy:
            logger.warning("No policy provided, using balanced")
            policy = self.get_policy("balanced")

        if not policy:
            return provider_scores

        # Apply constraints
        filtered_scores = []
        for provider_id, score in provider_scores:
            # Convert ProviderScore to dict for constraint checking
            score_dict = {
                "provider_id": provider_id,
                "latency_ms": score.latency_ms,
                "cost_estimate": score.cost_estimate,
                "reliability": score.reliability,
                "score": score.score,
                **score.metadata,
            }
            if self._meets_constraints(score_dict, policy.constraints):
                filtered_scores.append((provider_id, score))

        # Sort by composite score (already calculated in scoring)
        filtered_scores.sort(key=lambda x: x[1].score, reverse=True)

        return filtered_scores

    def _meets_constraints(
        self, score: Dict[str, Any], constraints: Dict[str, Any]
    ) -> bool:
        """Check if a provider score meets policy constraints.

        Args:
            score: Provider score dictionary
            constraints: Policy constraints

        Returns:
            True if constraints are met
        """
        for constraint_key, constraint_value in constraints.items():
            if constraint_key == "max_latency_ms":
                if score.get("latency_ms", 0) > constraint_value:
                    return False
            elif constraint_key == "max_cost_per_request":
                if score.get("cost_estimate", 0) > constraint_value:
                    return False
            elif constraint_key == "min_success_rate":
                if score.get("reliability", 0) < constraint_value:
                    return False
            # Add more constraint types as needed

        return True

    def save_policies(self):
        """Save current policies to configuration file."""
        try:
            config = {}
            for name, policy in self.policies.items():
                config[name] = {
                    "strategy": policy.strategy.value,
                    "weights": policy.weights,
                    "constraints": policy.constraints,
                    "fallbacks": policy.fallbacks,
                    "enabled": policy.enabled,
                }

            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved policies to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to save policies: {e}")


# Global policy manager instance
_policy_manager: Optional[PolicyManager] = None


def get_policy_manager() -> PolicyManager:
    """Get the global policy manager instance."""
    global _policy_manager
    if _policy_manager is None:
        _policy_manager = PolicyManager()
    return _policy_manager
