from fastapi import APIRouter

from .health.all_router import check_database_health, check_llm_provider_health
from .health.all_router import router as all_router
from .health.core_router import router as core_router
from .health.dashboard_router import router as dashboard_router

router = APIRouter(tags=["health"])

router.include_router(core_router, prefix="/health")
router.include_router(all_router, prefix="/health")
router.include_router(dashboard_router, prefix="/health")

__all__ = [
    "router",
    "check_database_health",
    "check_llm_provider_health",
]
