"""
Compatibility layer for the new routing subsystem.

Provides backward compatibility with the existing RoutingService interface
while delegating to the new unified routing subsystem.
"""

import logging
import os
from fastapi import Depends
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Generator
from sqlalchemy.orm import Session

from .routing_subsystem import get_routing_manager, RoutingManager
from .encryption import EncryptionService

# Import database dependency
try:
    from ..database import get_db
except ImportError:
    try:
        from database import get_db
    except ImportError:
        from backend.database import get_db

if TYPE_CHECKING:
    from backend.providers.base import InferenceRequest, InferenceResult
else:
    try:
        from providers.base import InferenceRequest, InferenceResult
    except Exception:
        from backend.providers.base import InferenceRequest, InferenceResult

logger = logging.getLogger(__name__)

# Get encryption key from environment
ROUTING_ENCRYPTION_KEY = os.getenv(
    "ROUTING_ENCRYPTION_KEY", "default-dev-key-change-me"
)


class RoutingServiceCompat:
    """Compatibility wrapper for the new routing subsystem.

    Maintains the same interface as the old RoutingService while using
    the new unified routing subsystem internally.
    """

    def __init__(self, db: Session, encryption_key: str):
        """Initialize compatibility wrapper.

        Args:
            db: Database session (kept for compatibility, may not be used)
            encryption_key: Encryption key (kept for compatibility)
        """
        self.db = db
        self.encryption_key = encryption_key
        self.encryption_service = EncryptionService(encryption_key)
        self.routing_manager: Optional[RoutingManager] = None

    async def initialize(self):
        """Initialize the routing subsystem."""
        if self.routing_manager is None:
            self.routing_manager = get_routing_manager()
            await self.routing_manager.start()
        logger.info("Routing service compatibility layer initialized")

    async def discover_providers(self) -> List[Dict[str, Any]]:
        """Discover all active providers and their capabilities.

        Returns:
            List of provider information dictionaries
        """
        if not self.routing_manager:
            await self.initialize()

        # Get system status from the new routing manager
        status = self.routing_manager.get_system_status()

        registry = getattr(self.routing_manager, "registry", None)

        def _normalize_provider_key(raw_key: Any) -> tuple[str, Optional[Any]]:
            """Return a stable provider_id string and (optionally) the provider object.

            In some environments we've seen provider keys accidentally be provider
            objects instead of strings; /chat/models should never 500 on that.
            """
            if isinstance(raw_key, str):
                provider_id = raw_key
                provider_obj = None
                if registry is not None:
                    try:
                        provider_obj = registry.get_provider(provider_id)
                    except Exception:
                        provider_obj = None

                    # Some registries key providers by config id rather than provider_id.
                    if provider_obj is None:
                        try:
                            for p in getattr(registry, "providers", {}).values():
                                pid = getattr(p, "provider_id", None)
                                if callable(pid):
                                    pid = pid()
                                if pid == provider_id:
                                    provider_obj = p
                                    break
                        except Exception:
                            provider_obj = None

                return provider_id, provider_obj

            # raw_key looks like an object; attempt to extract provider_id.
            provider_obj = raw_key
            provider_id = getattr(provider_obj, "provider_id", None)
            if callable(provider_id):
                provider_id = provider_id()
            if not isinstance(provider_id, str) or not provider_id:
                provider_id = provider_obj.__class__.__name__
            return provider_id, provider_obj

        providers: List[Dict[str, Any]] = []
        for raw_provider_id, provider_info in status.get("providers", {}).items():
            provider_id, provider_obj = _normalize_provider_key(raw_provider_id)

            # Provider-level capabilities (old shape expects a list).
            provider_caps = ["chat"]
            models: List[Dict[str, Any]] = []

            if provider_obj is not None:
                try:
                    caps = provider_obj.capabilities
                except Exception:
                    caps = {}
                if not isinstance(caps, dict):
                    caps = {}

                def _coerce_float(value: Any, default: float = 0.0) -> float:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return default

                supports_vision = bool(caps.get("supports_vision"))
                if supports_vision:
                    provider_caps = ["chat", "vision"]

                max_tokens = caps.get("max_tokens") or {}
                if not isinstance(max_tokens, dict):
                    max_tokens = {}
                cost_in = _coerce_float(caps.get("cost_per_token_input"))
                cost_out = _coerce_float(caps.get("cost_per_token_output"))

                models_list = caps.get("models") or []
                if not isinstance(models_list, list):
                    models_list = []

                for model_id in models_list:
                    model_caps = ["chat"]
                    if supports_vision:
                        model_caps.append("vision")
                    models.append(
                        {
                            "id": model_id,
                            "capabilities": model_caps,
                            "context_window": int(max_tokens.get(model_id, 0) or 0),
                            "pricing": {
                                "cost_per_token_input": cost_in,
                                "cost_per_token_output": cost_out,
                            },
                        }
                    )

            providers.append(
                {
                    "id": provider_id,
                    "name": provider_id,
                    "display_name": str(provider_id).replace("_", " ").title(),
                    "capabilities": provider_caps,
                    "models": models,
                    "priority": 1,
                    "is_active": provider_info.get("health") == "healthy",
                    "health_status": provider_info.get("health", "unknown"),
                    "metrics": provider_info.get("metrics", {}),
                }
            )

        return providers

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
        """Route a request to the best available provider.

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
        if not self.routing_manager:
            await self.initialize()

        import uuid

        request_id = str(uuid.uuid4())

        try:
            # Determine routing policy based on parameters
            policy_name = self._determine_policy(
                latency_priority, cost_budget, sla_target_ms
            )

            # Convert old format to new InferenceRequest
            inference_request = self._convert_to_inference_request(
                capability, requirements
            )

            # Use client_ip as client_key for rate limiting
            client_key = client_ip or user_id or "anonymous"

            # Route the request
            result = await self.routing_manager.route_request(
                request=inference_request,
                policy_name=policy_name,
                client_key=client_key,
            )

            # Convert result back to old format
            return self._convert_from_inference_result(result, request_id, capability)

        except Exception as e:
            logger.error(f"Routing request failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id,
            }

    def _determine_policy(
        self,
        latency_priority: Optional[str],
        cost_budget: Optional[float],
        sla_target_ms: Optional[float],
    ) -> str:
        """Determine routing policy based on request parameters.

        Args:
            latency_priority: Latency priority level
            cost_budget: Cost budget constraint
            sla_target_ms: SLA target

        Returns:
            Policy name to use
        """
        if latency_priority in ["ultra_low", "low"] or (
            sla_target_ms and sla_target_ms < 2000
        ):
            return "latency_first"
        elif cost_budget is not None and cost_budget < 0.01:
            return "cost_first"
        else:
            return "latency_first"  # Default policy

    def _convert_to_inference_request(
        self, capability: str, requirements: Optional[Dict[str, Any]]
    ) -> InferenceRequest:
        """Convert old request format to new InferenceRequest.

        Args:
            capability: Required capability
            requirements: Additional requirements

        Returns:
            InferenceRequest instance
        """
        reqs = requirements or {}

        # Extract messages from requirements - handle both old and new formats
        messages = reqs.get("messages", [])
        if not messages:
            # Handle old format where "message" is passed instead of "messages"
            message_content = reqs.get("message", "")
            if message_content:
                messages = [{"role": "user", "content": message_content}]
            else:
                # Create a default message if none provided
                messages = [{"role": "user", "content": "Hello"}]

        # Determine model and model family
        model = reqs.get("model", "")
        if not model:
            # Default model based on capability
            if capability == "chat":
                model = "gpt-3.5-turbo"
            else:
                model = "gpt-4"

        # Determine model family
        if "gpt" in model.lower():
            model_family = "openai"
        elif "claude" in model.lower():
            model_family = "anthropic"
        elif "llama" in model.lower() or "ollama" in model.lower():
            model_family = "ollama"
        else:
            model_family = "general"

        return InferenceRequest(
            messages=messages,
            model=model,
            model_family=model_family,
            max_tokens=reqs.get("max_tokens", 1000),
            temperature=reqs.get("temperature", 0.7),
            stream=reqs.get("stream", False),
        )

    def _convert_from_inference_result(
        self, result: InferenceResult, request_id: str, capability: str = "chat"
    ) -> Dict[str, Any]:
        """Convert InferenceResult back to old response format.

        Args:
            result: InferenceResult from new subsystem
            request_id: Request ID
            capability: The requested capability

        Returns:
            Dict in old response format
        """
        if hasattr(result, "success") and not result.success:
            return {
                "success": False,
                "error": getattr(result, "error", "Unknown error"),
                "request_id": request_id,
                "capability": capability,
            }

        # Extract provider info
        provider_id = getattr(result, "provider_id", "unknown")
        latency_ms = getattr(result, "latency_ms", 0)
        cost_usd = getattr(result, "cost_usd", 0.0)

        return {
            "success": True,
            "provider": {
                "id": provider_id,
                "name": provider_id,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            },
            "content": getattr(result, "content", ""),
            "usage": getattr(result, "usage", {}),
            "request_id": request_id,
            "capability": capability,
        }


# Create a global instance for backward compatibility
_routing_service_compat: Optional[RoutingServiceCompat] = None


def get_routing_service_compat(
    db: Session = Depends(get_db),
) -> RoutingServiceCompat:
    """Get the compatibility routing service instance.

    This is a FastAPI dependency that provides the routing service.
    """
    global _routing_service_compat
    if _routing_service_compat is None:
        _routing_service_compat = RoutingServiceCompat(db, ROUTING_ENCRYPTION_KEY)
    return _routing_service_compat


def get_routing_encryption_key() -> str:
    """Get the routing encryption key from environment.

    Returns:
        The encryption key string
    """
    return ROUTING_ENCRYPTION_KEY


# Alias for backward compatibility
RoutingService = RoutingServiceCompat
