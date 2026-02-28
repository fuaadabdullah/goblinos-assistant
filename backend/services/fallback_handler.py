"""
Stub fallback handler to keep routing imports functional in unit tests.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FallbackHandler:
    async def handle_emergency_routing(
        self, capability: str, requirements: Optional[Dict[str, Any]], request_id: str
    ) -> Dict[str, Any]:
        logger.warning(
            "FallbackHandler.handle_emergency_routing invoked (stub) for %s", capability
        )
        return {
            "success": True,
            "request_id": request_id,
            "capability": capability,
            "requirements": requirements or {},
            "provider": {"id": "emergency", "name": "emergency-fallback"},
            "is_fallback": True,
            "fallback_reason": "emergency_mode",
        }

    async def get_fallback_provider(self) -> Optional[Dict[str, Any]]:
        logger.debug("FallbackHandler.get_fallback_provider called (stub)")
        return {"id": "fallback", "display_name": "Default Fallback"}
