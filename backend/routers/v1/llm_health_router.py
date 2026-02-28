"""STATUS: V1 CONTRACT (FROZEN FOR NON-BREAKING EVOLUTION).

No new endpoints or schema changes should be introduced in this module.
Active development targets routers/v2.
Migration guide: docs/api-migration-v1-v2.md
"""

from ...health.llm_health import router as router

__all__ = ["router"]
