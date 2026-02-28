"""
Configuration processor service for handling provider configuration resolution.
Centralizes all configuration-related logic.
"""

from typing import Dict, Any, Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)


def get_routing_encryption_key() -> Optional[str]:
    """Get routing encryption key with dev-friendly behavior."""
    key = os.getenv("ROUTING_ENCRYPTION_KEY")
    if not key:
        message = (
            "ROUTING_ENCRYPTION_KEY environment variable must be set for chat "
            "routing functionality"
        )
        if os.getenv("ENVIRONMENT", "").lower() == "production":
            raise RuntimeError(message)
        logger.warning("%s (development mode: routing may be limited)", message)
    return key


def get_client_context(req: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract client context from request."""
    client_ip = req.client.host if req.client else None
    request_path = req.url.path if req.url else None
    user_id = getattr(req.state, "user_id", None) if hasattr(req, "state") else None
    return client_ip, request_path, user_id


def select_generation_params(
    request: Any, routing_result: Dict[str, Any]
) -> Tuple[float, int, float]:
    """Select generation parameters based on request and routing result."""
    recommended_params = routing_result.get("recommended_params", {})
    temperature = (
        request.temperature
        if request.temperature is not None
        else recommended_params.get("temperature", 0.2)
    )
    max_tokens = (
        request.max_tokens
        if request.max_tokens is not None
        else recommended_params.get("max_tokens", 512)
    )
    top_p = (
        request.top_p
        if request.top_p is not None
        else recommended_params.get("top_p", 0.95)
    )
    return temperature, max_tokens, top_p


def apply_system_prompt(
    messages: list, system_prompt: Optional[str]
) -> list:
    """Apply system prompt if provided and not already present."""
    if system_prompt and not any(msg["role"] == "system" for msg in messages):
        return [{"role": "system", "content": system_prompt}] + messages
    return messages


def build_requirements(
    request: Any,
    messages: list,
    gateway_result: Any,
) -> Dict[str, Any]:
    """Build requirements for routing based on request and gateway analysis."""
    requirements = {
        "messages": messages,
        "intent": request.intent or gateway_result.intent.value,
        "latency_target": request.latency_target,
        "context": request.context,
        "cost_priority": request.cost_priority,
    }

    if request.model:
        requirements["model"] = request.model

    return requirements


def prepare_messages(request: Any) -> list:
    """Prepare messages from request for processing."""
    return [{"role": msg.role, "content": msg.content} for msg in request.messages]


class ConfigProcessor:
    """
    Adapter class exposing configuration helper functions as methods for DI.
    """

    def get_routing_encryption_key(self) -> Optional[str]:
        return get_routing_encryption_key()

    def get_client_context(self, req: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return get_client_context(req)

    def select_generation_params(
        self, request: Any, routing_result: Dict[str, Any]
    ) -> Tuple[float, int, float]:
        return select_generation_params(request, routing_result)

    def apply_system_prompt(
        self, messages: list, system_prompt: Optional[str]
    ) -> list:
        return apply_system_prompt(messages, system_prompt)

    def build_requirements(
        self,
        request: Any,
        messages: list,
        gateway_result: Any,
    ) -> Dict[str, Any]:
        return build_requirements(request, messages, gateway_result)

    def prepare_messages(self, request: Any) -> list:
        return prepare_messages(request)
