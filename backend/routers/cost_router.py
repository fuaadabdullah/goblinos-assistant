# apps/goblin-assistant-root/backend/routers/cost_router.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, Any

from ..database import get_db
from ..models import Task

router = APIRouter()

@router.get("/cost-summary", response_model=Dict[str, Any], tags=["cost"])
async def get_cost_summary(
    user_id: str = Query(None),
    db: Session = Depends(get_db)
):
    """Get overall cost summary"""
    query = db.query(Task).filter(Task.status == "completed")
    if user_id:
        query = query.filter(Task.user_id == user_id)

    tasks = query.all()

    if not tasks:
        return {
            "total_cost": 0.0,
            "cost_by_provider": {},
            "cost_by_model": {},
        }

    # Some deployments use a Task model/table that doesn't track cost/tokens yet.
    # Use best-effort access so the dashboard doesn't 500.
    def task_cost(task: Task) -> float:
        try:
            value = getattr(task, "cost", 0.0)
        except Exception:
            value = 0.0
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    total_cost = sum(task_cost(task) for task in tasks)

    # Group costs by provider and model from task data
    cost_by_provider = {}
    cost_by_model = {}

    for task in tasks:
        provider = getattr(task, "provider", None) or "unknown"
        model = getattr(task, "model", None) or "unknown"
        cost = task_cost(task)

        cost_by_provider[provider] = cost_by_provider.get(provider, 0.0) + cost
        cost_by_model[model] = cost_by_model.get(model, 0.0) + cost

    return {
        "total_cost": round(total_cost, 4),
        "cost_by_provider": {k: round(v, 4) for k, v in cost_by_provider.items()},
        "cost_by_model": {k: round(v, 4) for k, v in cost_by_model.items()},
    }
