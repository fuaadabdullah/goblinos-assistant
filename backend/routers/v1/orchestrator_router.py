"""STATUS: V1 CONTRACT (FROZEN FOR NON-BREAKING EVOLUTION).

No new endpoints or schema changes should be introduced in this module.
Active development targets routers/v2.
Migration guide: docs/api-migration-v1-v2.md
"""

from ...orchestrator import orchestrator_router as router

__all__ = ["router"]
