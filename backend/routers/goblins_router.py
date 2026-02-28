# apps/goblin-assistant-root/backend/routers/goblins_router.py
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime, timezone
import random

from ..database import get_db
from ..models import Task

router = APIRouter()

@router.get("/goblins", response_model=List[Dict[str, Any]], tags=["goblins"])
async def get_goblins():
    """Get list of available goblins"""
    return [
        {
            "id": "docs-writer",
            "name": "docs-writer",
            "title": "Documentation Writer",
            "status": "available",
            "guild": "crafters",
        },
        {
            "id": "code-writer",
            "name": "code-writer",
            "title": "Code Writer",
            "status": "available",
            "guild": "crafters",
        },
    ]


@router.get("/goblins/{goblin}/history", response_model=List[Dict[str, Any]], tags=["goblins"])
async def get_goblin_history(
    goblin: str,
    limit: int = Query(10, ge=1, le=100),
    user_id: str = Query(None),
    db: Session = Depends(get_db)
):
    """Get task history for a specific goblin"""
    query = db.query(Task).filter(Task.goblin == goblin)
    if user_id:
        query = query.filter(Task.user_id == user_id)

    tasks = query.order_by(Task.created_at.desc()).limit(limit).all()

    history = []
    for task in tasks:
        # Older schemas/models may not store these KPIs yet.
        duration_ms = int(getattr(task, "duration_ms", 0) or 0)
        cost = float(getattr(task, "cost", 0.0) or 0.0)
        tokens = int(getattr(task, "tokens", 0) or 0)
        history.append(
            {
                "id": task.id,
                "goblin": task.goblin,
                "task": task.task,
                "response": task.result or "Task in progress",
                "timestamp": int(task.created_at.timestamp() * 1000),
                "kpis": f"duration_ms: {duration_ms}, cost: {cost}, tokens: {tokens}",
            }
        )
    return history


@router.get("/goblins/{goblin}/stats", response_model=Dict[str, Any], tags=["goblins"])
async def get_goblin_stats(
    goblin: str,
    user_id: str = Query(None),
    db: Session = Depends(get_db)
):
    """Get statistics for a specific goblin"""
    query = db.query(Task).filter(Task.goblin == goblin, Task.status == "completed")
    if user_id:
        query = query.filter(Task.user_id == user_id)

    tasks = query.all()

    if not tasks:
        return {
            "total_tasks": 0,
            "total_cost": 0.0,
            "avg_duration_ms": 0,
            "success_rate": 0.0,
            "last_used": None,
        }

    total_tasks = len(tasks)
    total_cost = sum(float(getattr(task, "cost", 0.0) or 0.0) for task in tasks)
    avg_duration = (
        sum(int(getattr(task, "duration_ms", 0) or 0) for task in tasks) / total_tasks
    )
    success_rate = 1.0  # All completed tasks are successful in our mock
    last_used = max(task.updated_at for task in tasks)
    last_used_timestamp = int(last_used.timestamp() * 1000)

    return {
        "total_tasks": total_tasks,
        "total_cost": round(total_cost, 4),
        "avg_duration_ms": int(avg_duration),
        "success_rate": success_rate,
        "last_used": last_used_timestamp,
    }
