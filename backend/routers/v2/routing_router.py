from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ...routing_router import routing_info as routing_info_v1
from ...services.imports import get_routing_service as get_unified_routing_service

router = APIRouter(prefix="/routing", tags=["routing"])


@router.get("/info")
async def routing_info_v2(service: Any = Depends(get_unified_routing_service)):
    """Version 2 routing info scaffold delegated to existing routing pipeline."""

    payload = await routing_info_v1(service=service)
    return JSONResponse(content=payload, headers={"X-API-Version": "v2"})


__all__ = ["router"]

