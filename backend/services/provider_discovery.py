"""
Lightweight provider discovery service placeholder.

The previous implementation was removed during refactors, but the routing
manager still imports this service. For unit tests that only exercise helper
logic we provide a minimal implementation that can be extended later.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ProviderDiscoveryService:
    """Discover available providers. Returns an empty list by default."""

    async def discover(self) -> List[Dict[str, Any]]:
        logger.debug("ProviderDiscoveryService.discover called (stub)")
        return []

    async def find_suitable_providers(
        self, capability: str, requirements: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Placeholder that would normally filter providers by capability/requirements.
        """
        logger.debug(
            "ProviderDiscoveryService.find_suitable_providers called (stub) "
            f"for capability={capability}"
        )
        return []
