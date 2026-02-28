"""Routing logging helpers."""

import logging
from typing import Any

try:
    from backend.models.provider import RoutingRequest
except ImportError:  # pragma: no cover - fallback for local module execution
    from models.provider import RoutingRequest

logger = logging.getLogger(__name__)


async def log_routing_request(
    db,
    request_id: str,
    capability: str,
    requirements: dict[str, Any] | None = None,
    selected_provider_id: int | None = None,
    success: bool = True,
    error_message: str | None = None,
) -> None:
    """Log a routing request asynchronously."""
    import asyncio

    def _sync_log():
        try:
            routing_request = RoutingRequest(
                request_id=request_id,
                capability=capability,
                requirements=requirements,
                selected_provider_id=selected_provider_id,
                success=success,
                error_message=error_message,
            )
            db.add(routing_request)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to log routing request: {e}")
            db.rollback()

    await asyncio.to_thread(_sync_log)
