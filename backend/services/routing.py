"""
Routing service for provider discovery, health monitoring, and intelligent task routing.

This service provides intelligent routing of AI inference requests to appropriate LLM providers
based on multi-factor scoring including health, performance, cost, SLA compliance, and capabilities.

See docs/backend/ROUTING_REFACTORING.md for complete architecture documentation.
"""

import uuid
import asyncio
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.provider import Provider, ProviderMetric, RoutingRequest

from ..providers import (
    OpenAIAdapter,
    AnthropicAdapter,
    GrokAdapter,
    DeepSeekAdapter,
    OllamaAdapter,
    LlamaCppAdapter,
    TinyLlamaAdapter,
    SiliconeflowAdapter,
    MoonshotAdapter,
    ElevenLabsAdapter,
    VertexAdapter,
)
from ..providers.base import InferenceRequest
from .encryption import EncryptionService
from .local_llm_routing import (
    select_model,
    get_system_prompt,
    get_routing_explanation,
    detect_intent,
    get_context_length,
    Intent,
    LatencyTarget,
)
from .autoscaling_service import AutoscalingService, FallbackLevel
from .latency_monitoring_service import LatencyMonitoringService
from . import routing_helpers
from .routing_config import RoutingConfig, get_default_config
from .provider_registry import ProviderRegistry
from .provider_scorer import ProviderScorer

logger = logging.getLogger(__name__)


class RoutingService:
    """Service for intelligent routing of AI tasks to appropriate providers.

    Refactored to use RoutingConfig for better testability and modularity.
    Configuration includes SLA targets, scoring weights, and adapter registry.
    """

    def __init__(
        self,
        db: Optional[Session] = None,
        async_db: Optional[AsyncSession] = None,
        encryption_key: Optional[str] = None,
        config: Optional[RoutingConfig] = None,
    ):
        """Initialize routing service.

        Args:
            db: Synchronous database session
            async_db: Async database session (preferred)
            encryption_key: Key for decrypting API keys (legacy, use config instead)
            config: Optional RoutingConfig instance. Defaults to environment-based config

        Note:
            Provide either db or async_db. If both provided, async_db takes precedence.
        """
        # Support both sync and async sessions
        if async_db is not None:
            self.async_db = async_db
            self.db = None  # Don't use sync session if async is provided
        elif db is not None:
            self.db = db
            self.async_db = None
        else:
            raise ValueError("Either db or async_db must be provided")

        # Use provided config or get default
        if config is None:
            if encryption_key:
                # Backward compatibility: create config from encryption key
                config = RoutingConfig.from_env(encryption_key)
            else:
                config = get_default_config()

        self.config = config
        config.validate()

        # Initialize services with config values
        self.encryption_service = EncryptionService(config.encryption_key)
        self.latency_monitor = LatencyMonitoringService()
        self.autoscaling_service = AutoscalingService()

        # Initialize provider scorer for intelligent ranking
        self.scorer = ProviderScorer(
            config=config,
            db=self.db,
            async_db=self.async_db,
            latency_monitor=self.latency_monitor,
        )

        # Initialize provider registry for adapter management
        self.provider_registry = ProviderRegistry(
            adapter_registry=config.adapter_registry,
            encryption_service=self.encryption_service,
        )

        # Expose config attributes for backward compatibility
        self.default_sla_targets = config.sla_targets
        self.cost_budget_weights = config.cost_budget_weights
        self.adapters = config.adapter_registry

    async def initialize(self):
        """Initialize async components"""
        await self.autoscaling_service.initialize()
        logger.info("Routing service initialized with autoscaling")

    async def discover_providers(self) -> List[Dict[str, Any]]:
        """Discover all active providers and their capabilities.

        Returns:
            List of provider information dictionaries
        """
        if self.async_db is not None:
            result = await self.async_db.execute(
                select(Provider).where(Provider.is_active == True)
            )
            providers = result.scalars().all()
        else:
            # Sync session path (uses thread pool)
            def _sync_query():
                return self.db.query(Provider).filter(Provider.is_active).all()

            providers = await asyncio.to_thread(_sync_query)

        result = []
        for provider in providers:
            # Use provider registry to initialize adapter
            adapter = await self.provider_registry.initialize_adapter(
                provider_name=provider.name,
                encrypted_key=provider.api_key_encrypted,
                plain_key=provider.api_key,
                base_url=getattr(provider, "base_url", None),
            )

            if not adapter:
                # Adapter initialization failed (logged in registry)
                continue

            # Get models from adapter
            models = await self.provider_registry.get_provider_models(adapter)

            # Resolve final base URL for response
            base_url = self.provider_registry.resolve_base_url(
                provider.name,
                getattr(provider, "base_url", None),
            )

            result.append(
                {
                    "id": provider.id,
                    "name": provider.name,
                    "display_name": provider.display_name,
                    "base_url": base_url,
                    "capabilities": provider.capabilities,
                    "models": models,
                    "priority": provider.priority,
                    "is_active": provider.is_active,
                }
            )

        return result

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
        """Route a request to the best available provider with autoscaling support.

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
        request_id = str(uuid.uuid4())

        try:
            # Handle autoscaling checks and emergency routing
            autoscaling_result = (
                await routing_helpers.handle_autoscaling_and_emergency_routing(
                    self,
                    capability,
                    requirements,
                    request_id,
                    client_ip,
                    user_id,
                    request_path,
                )
            )

            # If rate limit exceeded, return immediately
            if not autoscaling_result.get("success", True):
                return autoscaling_result

            # If emergency routing needed, handle it
            if autoscaling_result.get("emergency_routing"):
                return await self._route_emergency_request(
                    autoscaling_result["capability"],
                    autoscaling_result["requirements"],
                    autoscaling_result["request_id"],
                )

            # Continue with normal routing
            requirements = autoscaling_result.get("requirements", requirements)

            # Global auto-routing: evaluate all providers instead of preferring local LLMs
            # This ensures balanced evaluation across all available providers
            provider_result = (
                await routing_helpers.handle_provider_selection_and_fallback(
                    self,
                    capability,
                    requirements,
                    request_id,
                    sla_target_ms,
                    cost_budget,
                    latency_priority,
                )
            )

            return provider_result

        except Exception as e:
            return await routing_helpers.handle_routing_error(
                e, request_id, capability, requirements, self._log_routing_request
            )

    async def _find_suitable_providers(
        self, capability: str, requirements: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Find providers that can handle the given capability and requirements.

        For global auto-routing, we evaluate all providers that meet the basic requirements,
        ensuring comprehensive coverage across all available providers.

        Args:
            capability: Required capability
            requirements: Additional requirements

        Returns:
            List of suitable provider dictionaries
        """
        providers = await self.discover_providers()

        suitable = []
        for provider in providers:
            # Check if provider supports the capability
            if capability not in provider["capabilities"]:
                continue

            # Check additional requirements
            if requirements:
                if not self._check_requirements(provider, requirements):
                    continue

            suitable.append(provider)

        return suitable

    def _check_requirements(
        self, provider: Dict[str, Any], requirements: Dict[str, Any]
    ) -> bool:
        """Check if provider meets additional requirements.

        Args:
            provider: Provider information
            requirements: Requirements to check

        Returns:
            True if requirements are met
        """
        return routing_helpers.check_provider_requirements(provider, requirements)

    async def _score_providers(
        self,
        providers: List[Dict[str, Any]],
        capability: str,
        requirements: Optional[Dict[str, Any]] = None,
        sla_target_ms: Optional[float] = None,
        cost_budget: Optional[float] = None,
        latency_priority: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Score and rank providers based on health, performance, cost, and SLA compliance.

        Delegates to ProviderScorer for all scoring logic.

        Args:
            providers: List of provider candidates
            capability: Required capability
            requirements: Additional requirements
            sla_target_ms: SLA target response time in milliseconds
            cost_budget: Maximum cost per request in USD
            latency_priority: Latency priority level

        Returns:
            List of providers with scores, sorted by score descending
        """
        return await self.scorer.score_providers(
            providers=providers,
            capability=capability,
            requirements=requirements,
            sla_target_ms=sla_target_ms,
            cost_budget=cost_budget,
            latency_priority=latency_priority,
        )

    async def _should_use_fallback(
        self,
        providers: List[Dict[str, Any]],
        sla_target_ms: Optional[float] = None,
        latency_priority: Optional[str] = None,
    ) -> bool:
        """Check if we should use fallback to local models due to latency issues.

        Args:
            providers: Available providers
            sla_target_ms: SLA target response time
            latency_priority: Latency priority level

        Returns:
            True if fallback should be used
        """
        return await routing_helpers.should_use_latency_fallback(
            self.latency_monitor,
            providers,
            sla_target_ms,
            latency_priority,
            self.default_sla_targets,
        )

    async def _get_fallback_provider(self) -> Optional[Dict[str, Any]]:
        """Get fallback provider (local Mistral or Ollama 1b).

        Returns:
            Fallback provider info or None
        """
        return await routing_helpers.get_fallback_provider_info(
            self.db, self.encryption_service, self.adapters
        )

    async def _check_autoscaling(
        self,
        client_ip: Optional[str],
        user_id: Optional[str],
        request_path: Optional[str],
        capability: str,
    ) -> Dict[str, Any]:
        """Check autoscaling conditions and rate limits."""
        from .routing_helpers import check_autoscaling_conditions

        return await check_autoscaling_conditions(
            self.autoscaling_service, client_ip, user_id, request_path, capability
        )

    async def _route_emergency_request(
        self, capability: str, requirements: Optional[Dict[str, Any]], request_id: str
    ) -> Dict[str, Any]:
        """Route emergency requests with minimal functionality."""
        try:
            # For emergency mode, only allow basic health/auth endpoints
            if capability == "health":
                return {
                    "success": True,
                    "request_id": request_id,
                    "provider": {
                        "id": "emergency",
                        "name": "emergency",
                        "display_name": "Emergency Fallback",
                        "model": "basic",
                        "capabilities": ["health"],
                    },
                    "capability": capability,
                    "emergency_mode": True,
                }

            # For auth, try to find any available provider
            providers = await self.discover_providers()
            for provider in providers:
                if "auth" in provider.get("capabilities", []):
                    return {
                        "success": True,
                        "request_id": request_id,
                        "provider": provider,
                        "capability": capability,
                        "emergency_mode": True,
                    }

            # Fallback to cheap model for basic chat
            return {
                "success": True,
                "request_id": request_id,
                "provider": {
                    "id": "fallback",
                    "name": "ollama_gcp",
                    "display_name": "Cheap Fallback Model",
                    "model": self.autoscaling_service.cheap_fallback_model,
                    "capabilities": ["chat"],
                },
                "capability": capability,
                "emergency_mode": True,
                "fallback_model": True,
            }

        except Exception as e:
            logger.error(f"Emergency routing failed: {e}")
            return {
                "success": False,
                "error": "Emergency routing failed",
                "request_id": request_id,
            }

    async def _log_routing_request(
        self,
        request_id: str,
        capability: str,
        requirements: Optional[Dict[str, Any]] = None,
        selected_provider_id: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Log a routing request asynchronously.

        Args:
            request_id: Unique request ID
            capability: Requested capability
            requirements: Request requirements
            selected_provider_id: ID of selected provider
            success: Whether routing was successful
            error_message: Error message if failed
        """
        try:
            routing_request = RoutingRequest(
                request_id=request_id,
                capability=capability,
                requirements=requirements,
                selected_provider_id=selected_provider_id,
                success=success,
                error_message=error_message,
            )

            if self.async_db is not None:
                # Async database operation
                self.async_db.add(routing_request)
                await self.async_db.commit()
            else:
                # Sync session path (uses thread pool)
                def _sync_log():
                    try:
                        self.db.add(routing_request)
                        self.db.commit()
                    except Exception as e:
                        logger.error(f"Failed to log routing request: {e}")
                        self.db.rollback()

                await asyncio.to_thread(_sync_log)

        except Exception as e:
            logger.error(f"Failed to log routing request: {e}")
            if self.async_db is not None:
                await self.async_db.rollback()
