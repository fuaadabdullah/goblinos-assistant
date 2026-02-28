"""
RoutingManager Service for intelligent provider routing.

This service coordinates the various components involved in provider routing:
- Provider discovery
- Provider scoring
- Provider selection
- Fallback handling

This replaces the monolithic route_request method with a more
modular and testable architecture.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from .provider_discovery import ProviderDiscoveryService
from .provider_scorer import ProviderScorer
from .provider_selector import ProviderSelector
from .fallback_handler import FallbackHandler
from .autoscaling_service import AutoscalingService, FallbackLevel
from backend.gateway_service import GatewayService
from .local_llm_routing import (
    select_model,
    get_system_prompt,
    get_routing_explanation,
    detect_intent,
    get_context_length,
    Intent,
    LatencyTarget,
    estimate_token_count,
)
from models.provider import Provider

logger = logging.getLogger(__name__)


class RoutingManager:
    """Main manager for provider routing workflows."""

    def __init__(
        self,
        db: Session,
        provider_discovery: ProviderDiscoveryService,
        provider_scorer: ProviderScorer,
        provider_selector: ProviderSelector,
        fallback_handler: FallbackHandler,
        autoscaling_service: AutoscalingService,
        gateway_service: GatewayService,
    ):
        """Initialize the RoutingManager with all required services."""
        self.db = db
        self.provider_discovery = provider_discovery
        self.provider_scorer = provider_scorer
        self.provider_selector = provider_selector
        self.fallback_handler = fallback_handler
        self.autoscaling_service = autoscaling_service
        self.gateway_service = gateway_service

    async def route_request(
        self,
        capability: str,
        requirements: Optional[Dict[str, Any]] = None,
        sla_target_ms: Optional[float] = None,
        cost_budget: Optional[float] = None,
        latency_priority: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        request_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Route a request to the best available provider with autoscaling support.

        Args:
            capability: Required capability (e.g., "chat", "vision")
            requirements: Additional requirements for the request
            sla_target_ms: SLA target response time in milliseconds
            cost_budget: Maximum cost per request in USD
            latency_priority: Latency priority ('ultra_low', 'low', 'medium', 'high')
            client_ip: Client IP for rate limiting
            user_id: User ID for rate limiting
            request_path: Request path for emergency endpoint detection

        Returns:
            Dict with routing decision and provider info
        """
        request_id = self._generate_request_id()

        try:
            # Step 1: Check autoscaling and rate limiting
            autoscaling_result = await self._check_autoscaling(
                client_ip, user_id, request_path, capability
            )

            if not autoscaling_result["allowed"]:
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "fallback_level": autoscaling_result["fallback_level"].value,
                    "retry_after": autoscaling_result["retry_after"],
                    "request_id": request_id,
                }

            fallback_level = autoscaling_result["fallback_level"]

            # Step 2: Handle emergency mode or emergency endpoints
            if (
                fallback_level == FallbackLevel.EMERGENCY
                or autoscaling_result["emergency_endpoint"]
            ):
                return await self.fallback_handler.handle_emergency_routing(
                    capability, requirements, request_id
                )

            # Step 3: Check if this is a chat request that can be handled by local LLMs
            if capability == "chat" and requirements:
                local_routing = await self._try_local_llm_routing(requirements)
                if local_routing:
                    return {
                        "success": True,
                        "request_id": request_id,
                        "provider": local_routing["provider"],
                        "capability": capability,
                        "requirements": requirements,
                        "routing_explanation": local_routing.get("explanation"),
                        "recommended_params": local_routing.get("params"),
                        "system_prompt": local_routing.get("system_prompt"),
                        "local_llm_routing": True,
                    }

            # Step 4: Discover suitable providers
            candidates = await self.provider_discovery.find_suitable_providers(
                capability, requirements
            )

            if not candidates:
                return {
                    "success": False,
                    "error": f"No providers available for capability: {capability}",
                    "request_id": request_id,
                }

            # Step 5: Check if we should use fallback due to latency issues
            use_fallback = await self._should_use_fallback(
                candidates, sla_target_ms, latency_priority
            )

            if use_fallback:
                fallback_provider = await self.fallback_handler.get_fallback_provider()
                if fallback_provider:
                    logger.info(
                        f"Using fallback provider: {fallback_provider['display_name']}"
                    )

                    return {
                        "success": True,
                        "request_id": request_id,
                        "provider": fallback_provider,
                        "capability": capability,
                        "requirements": requirements,
                        "is_fallback": True,
                        "fallback_reason": "latency_sla_violation",
                    }

            # Step 6: Score and rank providers
            scored_providers = await self.provider_scorer.score_providers(
                candidates,
                capability,
                requirements,
                sla_target_ms,
                cost_budget,
                latency_priority,
            )

            if not scored_providers:
                return {
                    "success": False,
                    "error": "No healthy providers available",
                    "request_id": request_id,
                }

            # Step 7: Select the best provider
            selected_provider = await self.provider_selector.select_best_provider(
                scored_providers, requirements
            )

            return {
                "success": True,
                "request_id": request_id,
                "provider": selected_provider,
                "capability": capability,
                "requirements": requirements or {},
                "routing_metadata": {
                    "score": selected_provider.get("score"),
                    "selection_reason": "highest_score",
                    "candidates_considered": len(scored_providers),
                },
            }

        except Exception as e:
            logger.error(f"Routing failed for capability {capability}: {e}")

            return {
                "success": False,
                "error": str(e),
                "request_id": request_id,
            }

    async def _check_autoscaling(
        self,
        client_ip: Optional[str],
        user_id: Optional[str],
        request_path: Optional[str],
        capability: str,
    ) -> Dict[str, Any]:
        """Check autoscaling conditions and rate limits."""
        try:
            # Check if emergency mode
            if await self.autoscaling_service.is_emergency_mode():
                return {
                    "allowed": True,
                    "fallback_level": FallbackLevel.EMERGENCY,
                    "emergency_endpoint": True,
                    "retry_after": None,
                }

            # Check if emergency endpoint
            if request_path and await self.autoscaling_service.is_emergency_endpoint(
                request_path
            ):
                return {
                    "allowed": True,
                    "fallback_level": FallbackLevel.NORMAL,
                    "emergency_endpoint": True,
                    "retry_after": None,
                }

            # Check rate limiting
            (
                allowed,
                fallback_level,
                metadata,
            ) = await self.autoscaling_service.check_rate_limit(
                client_ip or "unknown", user_id
            )

            return {
                "allowed": allowed,
                "fallback_level": fallback_level,
                "emergency_endpoint": False,
                "retry_after": metadata.get("cooldown_until"),
            }

        except Exception as e:
            logger.error(f"Autoscaling check failed: {e}")
            # Allow request but log error
            return {
                "allowed": True,
                "fallback_level": FallbackLevel.NORMAL,
                "emergency_endpoint": False,
                "retry_after": None,
            }

    async def _try_local_llm_routing(
        self, requirements: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Try to route request to local LLM based on intelligent routing rules."""
        try:
            # Extract routing parameters
            messages = requirements.get("messages", [])
            if not messages:
                return None

            # Get optional routing hints
            intent = requirements.get("intent")
            if intent and isinstance(intent, str):
                try:
                    intent = Intent(intent)
                except ValueError:
                    intent = None

            latency_target = requirements.get("latency_target", "medium")
            if isinstance(latency_target, str):
                try:
                    latency_target = LatencyTarget(latency_target)
                except ValueError:
                    latency_target = LatencyTarget.MEDIUM

            context_provided = requirements.get("context")
            cost_priority = requirements.get("cost_priority", False)

            # Check autoscaling service for rate limiting and fallback
            client_ip = requirements.get("client_ip", "unknown")
            user_id = requirements.get("user_id")

            (
                allowed,
                fallback_level,
                rate_limit_metadata,
            ) = await self.autoscaling_service.check_rate_limit(
                client_ip=client_ip, user_id=user_id
            )

            if not allowed:
                logger.warning(
                    f"Request denied due to rate limiting: {rate_limit_metadata}"
                )
                return {
                    "error": "Rate limit exceeded",
                    "fallback_level": fallback_level.value,
                    "retry_after": 60,
                    "request_id": requirements.get("request_id"),
                }

            force_cheap_fallback = fallback_level == FallbackLevel.CHEAP_MODEL

            # Select model using routing logic
            model_id, params = select_model(
                messages=messages,
                intent=intent,
                latency_target=latency_target,
                context_provided=context_provided,
                cost_priority=cost_priority,
                force_cheap_fallback=force_cheap_fallback,
            )

            # Find Ollama provider
            def _sync_find_provider():
                return (
                    self.db.query(Provider)
                    .filter(Provider.name == "ollama_gcp", Provider.is_active)
                    .first()
                )

            ollama_provider = await asyncio.to_thread(_sync_find_provider)

            if not ollama_provider:
                logger.warning("Ollama provider not found or not active")
                return None

            # Get system prompt
            detected_intent = intent or detect_intent(messages)
            system_prompt = get_system_prompt(detected_intent)

            # Get routing explanation
            context_length = get_context_length(messages)
            if context_provided:
                from .local_llm_routing import estimate_token_count

                context_length += estimate_token_count(context_provided)

            explanation = get_routing_explanation(
                model_id, detected_intent, context_length, latency_target
            )

            # Build provider info
            provider_info = {
                "id": ollama_provider.id,
                "name": ollama_provider.name,
                "display_name": ollama_provider.display_name,
                "base_url": ollama_provider.base_url,
                "model": model_id,
                "capabilities": ollama_provider.capabilities,
                "priority": ollama_provider.priority,
            }

            return {
                "provider": provider_info,
                "provider_id": ollama_provider.id,
                "params": params,
                "system_prompt": system_prompt,
                "explanation": explanation,
                "intent": detected_intent.value,
                "context_length": context_length,
            }

        except Exception as e:
            logger.error(f"Local LLM routing failed: {e}")
            return None

    async def _should_use_fallback(
        self,
        providers: List[Dict[str, Any]],
        sla_target_ms: Optional[float] = None,
        latency_priority: Optional[str] = None,
    ) -> bool:
        """Check if we should use fallback to local models due to latency issues."""
        # If no SLA target or low latency priority, don't fallback
        if not sla_target_ms or latency_priority in ["medium", "high"]:
            return False

        # Check if any providers are meeting SLA targets
        sla_target = sla_target_ms or self.default_sla_targets.get(
            latency_priority, 2000
        )

        compliant_providers = 0
        total_checked = 0

        for provider in providers[:3]:  # Check top 3 providers
            if provider.get("models"):
                model_name = provider["models"][0]["id"]
                try:
                    sla_check = await self.latency_monitor.check_sla_compliance(
                        provider["name"], model_name, sla_target
                    )
                    total_checked += 1
                    if sla_check.get("compliant", False):
                        compliant_providers += 1
                except Exception as e:
                    logger.warning(f"Failed to check SLA for fallback: {e}")

        # If less than 50% of providers meet SLA, use fallback
        if total_checked > 0 and (compliant_providers / total_checked) < 0.5:
            logger.info(
                f"Using fallback: only {compliant_providers}/{total_checked} providers meet SLA target of {sla_target}ms"
            )
            return True

        return False

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        import uuid

        return f"route_{uuid.uuid4().hex[:16]}"

    async def get_routing_info(self) -> Dict[str, Any]:
        """Get information about the intelligent routing system."""
        return {
            "routing_system": "intelligent",
            "version": "1.0",
            "factors": {
                "intent": {
                    "description": "Detected or explicit intent (code-gen, creative, rag, chat, etc.)",
                    "options": [
                        "code-gen",
                        "creative",
                        "explain",
                        "summarize",
                        "rag",
                        "retrieval",
                        "chat",
                        "classification",
                        "status",
                        "translation",
                    ],
                    "auto_detect": True,
                },
                "latency_target": {
                    "description": "Target latency for response",
                    "options": ["ultra_low", "low", "medium", "high"],
                    "default": "medium",
                },
                "context_length": {
                    "description": "Length of the conversation context",
                    "thresholds": {
                        "short": "< 2000 tokens",
                        "medium": "2000-8000 tokens",
                        "long": "> 8000 tokens (uses qwen2.5:3b with 32K window)",
                    },
                },
                "cost_priority": {
                    "description": "Prioritize cost over quality",
                    "default": False,
                    "effect": "Routes to smaller, faster models when enabled",
                },
            },
            "models": {
                "gemma:2b": {
                    "size": "1.7GB",
                    "context": "8K tokens",
                    "latency": "5-8s",
                    "best_for": ["ultra_fast", "classification", "status_checks"],
                    "params": {"temperature": 0.0, "max_tokens": 40},
                },
                "phi3:3.8b": {
                    "size": "2.2GB",
                    "context": "4K tokens",
                    "latency": "10-12s",
                    "best_for": ["low_latency_chat", "conversational_ui", "quick_qa"],
                    "params": {"temperature": 0.15, "max_tokens": 128},
                },
                "qwen2.5:3b": {
                    "size": "1.9GB",
                    "context": "32K tokens",
                    "latency": "14s",
                    "best_for": [
                        "long_context",
                        "multilingual",
                        "rag",
                        "document_retrieval",
                    ],
                    "params": {"temperature": 0.0, "max_tokens": 1024},
                },
                "mistral:7b": {
                    "size": "4.4GB",
                    "context": "8K tokens",
                    "latency": "14-15s",
                    "best_for": [
                        "high_quality",
                        "code_generation",
                        "creative_writing",
                        "explanations",
                    ],
                    "params": {"temperature": 0.2, "max_tokens": 512},
                },
            },
            "cost": {
                "per_request": "$0 (self-hosted)",
                "monthly_infrastructure": "$15-20",
                "savings_vs_cloud": "86-92% ($110-240/month)",
            },
        }
