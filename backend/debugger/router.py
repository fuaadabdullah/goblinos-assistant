from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
import logging

from .model_router import ModelRouter

router = APIRouter(prefix="/debugger", tags=["debugger"])
logger = logging.getLogger(__name__)
model_router = ModelRouter()


@router.post("/suggest")
async def get_debug_suggestion(
    task: str = Body(
        ..., description="Debug task type (e.g., 'quick_fix', 'summarize_trace')"
    ),
    context: Dict[str, Any] = Body(..., description="Context data for the debug task"),
):
    # Validate task input
    if not isinstance(task, str) or not task.strip():
        raise HTTPException(status_code=400, detail="Task must be a non-empty string")

    if not isinstance(context, dict):
        raise HTTPException(status_code=400, detail="Context must be a dictionary")

    try:
        route = model_router.choose_model(task, context)
    except RuntimeError as e:
        logger.error(f"Model routing failed for task '{task}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

    payload = {
        "task": task,
        "context": context,
        "metadata": {"source": "goblin-assistant", "timestamp": "2025-11-25"},
    }

    try:
        result = await model_router.call_model(route, payload)
    except Exception as e:
        logger.exception(f"Model call failed for {route.model_name}: {e}")
        raise HTTPException(status_code=502, detail=f"Model call failed: {str(e)}")

    # Extract suggestion with fallback
    suggestion = (
        result.get("suggestion") or result.get("text") or result.get("response") or ""
    )

    # Add confidence if available
    confidence = result.get("confidence")

    response = {
        "model": route.model_name,
        "suggestion": suggestion,
        "confidence": confidence,
        "task": task,
        "timestamp": "2025-11-25",
    }

    # Include raw result for debugging (optional)
    if result:
        response["raw"] = result

    return response
