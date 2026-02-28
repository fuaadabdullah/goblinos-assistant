"""
Utility service containing helper functions and common utilities.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def _coalesce(*values: Optional[str]) -> Optional[str]:
    """Return the first non-empty value."""
    for value in values:
        if value:
            return value
    return None


def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    """Normalize base URLs to avoid duplicate /v1 segments."""
    if not base_url:
        return None
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def _resolve_provider_kind(provider_name: Optional[str]) -> Optional[str]:
    """Map provider names to local provider kinds."""
    name = (provider_name or "").lower()
    if "ollama" in name:
        return "ollama"
    if "llamacpp" in name or "llama.cpp" in name:
        return "llamacpp"
    return None


def build_error_response(
    error_type: str,
    error_message: str,
    request_id: Optional[str] = None,
    status_code: int = 500,
) -> Dict[str, Any]:
    """Build a standardized error response."""
    from datetime import datetime

    return {
        "error": {
            "type": error_type,
            "message": error_message,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
        }
    }


def validate_api_key(api_key: Optional[str], provider_name: str) -> str:
    """Validate and return API key for a provider."""
    if not api_key:
        raise ValueError(f"API key not configured for provider {provider_name}")
    return api_key


def get_environment_variable(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default."""
    import os

    return os.getenv(var_name, default)


def is_production() -> bool:
    """Check if running in production environment."""
    import os

    return os.getenv("ENVIRONMENT", "").lower() == "production"


def get_provider_config(provider_name: str) -> Dict[str, Any]:
    """Get provider configuration from environment variables."""
    import os

    config = {
        "base_url": os.getenv(f"{provider_name.upper()}_BASE_URL"),
        "api_key": os.getenv(f"{provider_name.upper()}_API_KEY"),
        "model": os.getenv(f"{provider_name.upper()}_MODEL"),
    }

    # Remove None values
    return {k: v for k, v in config.items() if v is not None}


def format_duration(milliseconds: float) -> str:
    """Format duration in milliseconds to human-readable string."""
    if milliseconds < 1000:
        return f"{milliseconds:.0f}ms"
    elif milliseconds < 60000:
        return f"{milliseconds / 1000:.1f}s"
    else:
        minutes = milliseconds / 60000
        return f"{minutes:.1f}min"


def sanitize_text(text: str) -> str:
    """Sanitize text for logging and display."""
    if not text:
        return ""
    # Remove or replace problematic characters
    return text.replace("\x00", "").replace("\n", " ").replace("\r", " ").strip()


def get_token_estimate(text: str) -> int:
    """Estimate token count for text (rough estimation)."""
    # Simple estimation: ~4 characters per token
    return max(1, len(text) // 4)


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def get_nested_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get nested value from dictionary using dot notation path."""
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
    """Set nested value in dictionary using dot notation path."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


async def record_latency_metric(
    provider_name: str,
    model_name: str,
    response_time_ms: float,
    tokens_used: int,
    success: bool,
    error_type: Optional[str] = None,
) -> None:
    """
    Shared latency metric helper used across services.

    This mirrors the logic in verification_processor while keeping it reusable.
    """
    try:
        from .latency_monitoring_service import LatencyMonitoringService

        latency_monitor = LatencyMonitoringService()
        await latency_monitor.record_metric(
            provider_name=provider_name,
            model_name=model_name,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            success=success,
            error_type=error_type,
        )
    except Exception as e:
        logger.warning(f"Failed to record latency metric: {e}")


async def _record_latency_metric(
    provider_name: str,
    model_name: str,
    response_time_ms: float,
    tokens_used: int,
    success: bool,
    error_type: Optional[str] = None,
) -> None:
    """Compatibility wrapper used by chat_controller tests."""
    await record_latency_metric(
        provider_name, model_name, response_time_ms, tokens_used, success, error_type
    )


def get_provider_class(provider_name: str):
    """
    Resolve a provider adapter class by name.
    Returns None in this stub to avoid heavy imports during unit tests.
    """
    try:
        from providers.registry import get_provider_registry

        registry = get_provider_registry()
        return registry.get(provider_name)
    except Exception:
        return None


def get_provider_model_config(provider_name: str) -> Dict[str, Any]:
    """
    Retrieve model configuration for a provider.
    Stub returns an empty dict to keep tests lightweight.
    """
    return {}


class Utils:
    """
    Convenience wrapper so the utilities can be accessed via a service instance.
    """

    def build_error_response(
        self,
        error_type: str,
        error_message: str,
        request_id: Optional[str] = None,
        status_code: int = 500,
    ) -> Dict[str, Any]:
        return build_error_response(error_type, error_message, request_id, status_code)

    async def record_latency_metric(
        self,
        provider_name: str,
        model_name: str,
        response_time_ms: float,
        tokens_used: int,
        success: bool,
        error_type: Optional[str] = None,
    ) -> None:
        await record_latency_metric(
            provider_name, model_name, response_time_ms, tokens_used, success, error_type
        )
