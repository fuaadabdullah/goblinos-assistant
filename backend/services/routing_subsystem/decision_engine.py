"""
Decision engine for provider selection and routing logic.

Composes scoring functions and applies policies to select optimal providers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple
from dataclasses import dataclass

from .scoring import ProviderScore, score_provider
from .policies import PolicyManager, RoutingPolicy
from .cache import get_routing_cache
from .provider_health import get_provider_health_monitor

if TYPE_CHECKING:
    # Static analysis sees the canonical path
    from backend.providers.registry import get_provider_registry
    from backend.providers.base import InferenceRequest, ProviderBase
else:
    # Runtime uses available import path
    try:
        from providers.registry import get_provider_registry
        from providers.base import InferenceRequest, ProviderBase
    except ImportError:
        from backend.providers.registry import get_provider_registry
        from backend.providers.base import InferenceRequest, ProviderBase

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Routing decision result."""

    provider_id: str
    score: ProviderScore
    reason: str
    fallback_providers: List[str]
    cache_hit: bool = False


class DecisionEngine:
    """Engine for making routing decisions."""

    def __init__(self):
        """Initialize decision engine."""
        self.registry = get_provider_registry()
        self.policy_manager = PolicyManager()
        self.cache = get_routing_cache()
        self.health_monitor = get_provider_health_monitor()

    def select_provider(
        self,
        request: InferenceRequest,
        policy_name: str = "latency_first",
        use_cache: bool = True,
    ) -> RoutingDecision:
        """Select the best provider for a request.

        Args:
            request: Inference request
            policy_name: Routing policy to apply
            use_cache: Whether to use decision caching

        Returns:
            Routing decision
        """
        # Generate request hash for caching
        request_hash = self._hash_request(request, policy_name)

        # Check cache first
        if use_cache:
            cached_decision = self.cache.get_cached_decision("global", request_hash)
            if cached_decision:
                return RoutingDecision(
                    provider_id=cached_decision.provider_id,
                    score=cached_decision.score,
                    reason=f"Cached decision: {cached_decision.reason}",
                    fallback_providers=cached_decision.fallback_providers,
                    cache_hit=True,
                )

        # Get available providers - for now, get all available providers
        # TODO: Filter by model family or capability when providers report this properly
        available_providers = self.registry.get_available_providers()

        if not available_providers:
            raise ValueError("No providers available")

        # Deterministic low-latency fast path for lightweight greetings/single-turn prompts.
        if self._is_simple_prompt_request(request):
            fast_path_order = [
                "ollama_gcp",
                "llamacpp_gcp",
                "ollama",
                "openai",
                "anthropic",
            ]
            available_by_id = {provider.provider_id: provider for provider in available_providers}
            for provider_id in fast_path_order:
                provider = available_by_id.get(provider_id)
                if not provider:
                    continue
                score = self._score_provider(provider_id, provider, request)
                fallback_providers = [
                    pid for pid in fast_path_order if pid in available_by_id and pid != provider_id
                ]
                return RoutingDecision(
                    provider_id=provider_id,
                    score=score,
                    reason="Selected via fast_path_simple_prompt",
                    fallback_providers=fallback_providers,
                    cache_hit=False,
                )

        # Score all providers
        scored_providers = []
        for provider in available_providers:
            provider_id = provider.provider_id
            score = self._score_provider(provider_id, provider, request)
            scored_providers.append((provider_id, score))

        # Apply routing policy
        policy = self.policy_manager.get_policy(policy_name)
        if not policy:
            logger.warning(f"Policy {policy_name} not found, using default")
            policy = self.policy_manager.get_policy("latency_first")

        selected_providers = self.policy_manager.apply_policy(scored_providers, policy)

        if not selected_providers:
            raise ValueError(f"No providers passed policy filters: {policy_name}")

        # Select primary provider
        primary_provider_id, primary_score = selected_providers[0]

        # Get fallback providers
        fallback_providers = [pid for pid, _ in selected_providers[1:]]

        decision = RoutingDecision(
            provider_id=primary_provider_id,
            score=primary_score,
            reason=self._generate_reason(primary_score, policy),
            fallback_providers=fallback_providers,
        )

        # Cache decision
        if use_cache:
            self.cache.cache_decision("global", request_hash, decision)

        return decision

    def _is_simple_prompt_request(self, request: InferenceRequest) -> bool:
        """Heuristic for short conversational prompts requiring low latency."""
        if not request.messages:
            return False
        user_messages = [m for m in request.messages if m.get("role") == "user"]
        if len(user_messages) != 1:
            return False
        if len(request.messages) > 3:
            return False
        content = (user_messages[0].get("content") or "").strip()
        if not content:
            return False
        return len(content) <= 32

    def _score_provider(
        self, provider_id: str, provider: ProviderBase, request: InferenceRequest
    ) -> ProviderScore:
        """Score a provider for a request.

        Args:
            provider_id: Provider identifier
            provider: Provider instance
            request: Inference request

        Returns:
            Provider score
        """
        # Get health status
        health_metrics = self.health_monitor.get_provider_health(provider_id)
        health_status = health_metrics.get("health_status", "unknown")

        # Get cached metrics
        metrics = self.cache.get_provider_metrics(provider_id)

        # Estimate cost
        try:
            cost_estimate = provider.estimate_cost(request)
        except Exception as e:
            logger.warning(f"Cost estimation failed for {provider_id}: {e}")
            cost_estimate = 0.01  # Default fallback

        # Build provider metadata for scoring
        provider_meta = {
            "provider_id": provider_id,
            "health_status": health_status,
            "avg_latency_ms": metrics.get("avg_latency_ms", 1000),
            "cost_estimate": cost_estimate,
            "reliability_score": metrics.get("reliability_score", 0.5),
            **health_metrics,
            **metrics,
        }

        # Build request context
        request_context = {
            "model": request.model,
            "model_family": request.model_family,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages_count": len(request.messages) if request.messages else 0,
        }

        # Score provider
        return score_provider(provider_meta, request_context)

    def _hash_request(self, request: InferenceRequest, policy_name: str) -> str:
        """Generate hash for request caching.

        Args:
            request: Inference request
            policy_name: Policy name

        Returns:
            Request hash
        """
        # Create hashable representation
        hash_data = {
            "model_family": request.model_family,
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages_count": len(request.messages) if request.messages else 0,
            "policy": policy_name,
        }

        # Hash the data
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def _generate_reason(self, score: ProviderScore, policy: RoutingPolicy) -> str:
        """Generate human-readable reason for decision.

        Args:
            score: Provider score
            policy: Applied policy

        Returns:
            Reason string
        """
        reasons = []

        if policy.name == "latency_first":
            reasons.append(
                f"Lowest latency: {score.metadata.get('latency_score', 0):.2f}"
            )
        elif policy.name == "cost_first":
            reasons.append(f"Lowest cost: ${score.metadata.get('cost_score', 0):.4f}")
        elif policy.name == "reliability_first":
            reasons.append(
                f"Highest reliability: {score.metadata.get('reliability_score', 0):.2f}"
            )

        if score.metadata.get("health_penalty", 0) > 0:
            reasons.append(
                f"Health penalty: -{score.metadata.get('health_penalty', 0):.2f}"
            )

        if score.metadata.get("load_penalty", 0) > 0:
            reasons.append(
                f"Load penalty: -{score.metadata.get('load_penalty', 0):.2f}"
            )

        return f"Selected via {policy.name}: {', '.join(reasons)}"

    def get_provider_rankings(
        self, request: InferenceRequest, policy_name: str = "latency_first"
    ) -> List[Tuple[str, ProviderScore]]:
        """Get ranked list of all providers for a request.

        Args:
            request: Inference request
            policy_name: Routing policy

        Returns:
            List of (provider_id, score) tuples, ranked by policy
        """
        # Get available providers
        capability = request.model_family or "chat"
        available_providers = self.registry.get_providers_by_capability(capability)

        # Score all providers
        scored_providers = []
        for provider in available_providers:
            provider_id = provider.provider_id
            score = self._score_provider(provider_id, provider, request)
            scored_providers.append((provider_id, score))

        # Apply policy for ranking
        policy = self.policy_manager.get_policy(policy_name)
        if policy:
            return self.policy_manager.apply_policy(scored_providers, policy)
        else:
            # Default ranking by total score
            return sorted(
                scored_providers, key=lambda x: x[1].total_score, reverse=True
            )

    def validate_request(self, request: InferenceRequest) -> List[str]:
        """Validate a request for routing.

        Args:
            request: Inference request

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not request.messages:
            errors.append("Request must contain messages")

        if request.max_tokens and request.max_tokens <= 0:
            errors.append("max_tokens must be positive")

        if request.temperature is not None and not (0 <= request.temperature <= 2):
            errors.append("temperature must be between 0 and 2")

        # Check if any providers are available
        available_providers = self.registry.get_available_providers()

        if not available_providers:
            errors.append("No providers available")

        return errors


# Global decision engine instance
_engine: Optional[DecisionEngine] = None


def get_decision_engine() -> DecisionEngine:
    """Get the global decision engine instance."""
    global _engine
    if _engine is None:
        _engine = DecisionEngine()
    assert _engine is not None
    return _engine
