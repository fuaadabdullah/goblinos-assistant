"""
Celery-based Task Queue - Replacement for RQ functionality

This module provides the same interface as the original task_queue.py
but uses Celery for enterprise-grade task management with:
- Priority queues and routing
- Result storage and tracking
- Comprehensive logging and monitoring
- Retry logic and error handling
- Scheduled task execution
"""

import os
import json
import time
import logging
import redis
from typing import Dict, Any, List, Optional

from celery_app import app as celery_app, GoblinTask
from celery.result import AsyncResult

logger = logging.getLogger(__name__)

# Redis client for direct operations (logs, artifacts, etc.)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL)


@celery_app.task(bind=True, base=GoblinTask)
def process_task_celery(self, task_id: str, payload: Dict[str, Any]):
    """
    Celery task that replaces RQ's process_task function.
    This is the main task processor that will be called by Celery workers.
    """
    try:
        # Log task start
        add_task_log(task_id, "info", f"Task {task_id} started by Celery worker")

        # Set task as running
        set_task_running(task_id)

        # Process the task (this replaces the original process_task logic)
        # Simulate doing work - replace with actual task processing
        time.sleep(2)

        # Create artifact (example)
        artifact = {
            "name": "result.txt",
            "contents": f"Task {task_id} executed via Celery",
            "timestamp": time.time(),
        }
        add_task_artifact(task_id, artifact)

        add_task_log(task_id, "info", f"Task {task_id} completed successfully")

        # Mark as completed
        set_task_completed(task_id, "success")

        return {"status": "success", "task_id": task_id, "artifacts": [artifact]}

    except Exception as exc:
        # Log error
        add_task_log(task_id, "error", f"Task {task_id} failed: {str(exc)}")

        # Re-raise to let Celery handle retry/error logic
        raise self.retry(countdown=60, exc=exc)


def enqueue_task(task_id: str, payload: Dict[str, Any], queue: str = "default") -> str:
    """
    Enqueue a task using Celery instead of RQ.

    Args:
        task_id: Unique task identifier
        payload: Task payload data
        queue: Celery queue to use (default, high_priority, low_priority, batch, scheduled)

    Returns:
        Task ID
    """
    # Store initial task metadata in Redis (for compatibility with existing APIs)
    key = f"task:{task_id}"
    redis_client.hset(
        key,
        mapping={
            "status": "queued",
            "created_at": str(time.time()),
            "payload": json.dumps(payload),
            "queue": queue,
        },
    )

    # Enqueue the task with Celery
    task_result = process_task_celery.apply_async(
        args=[task_id, payload],
        queue=queue,
        task_id=task_id,  # Use our task_id as Celery task ID
    )

    logger.info(
        f"Enqueued task {task_id} to Celery queue '{queue}' with Celery ID {task_result.id}"
    )

    return task_id


def set_task_running(task_id: str):
    """Mark task as running (compatible with existing API)"""
    key = f"task:{task_id}"
    redis_client.hset(
        key, mapping={"status": "running", "started_at": str(time.time())}
    )


def set_task_completed(task_id: str, result: str):
    """Mark task as completed (compatible with existing API)"""
    key = f"task:{task_id}"
    redis_client.hset(
        key,
        mapping={
            "status": "completed",
            "result": result,
            "completed_at": str(time.time()),
        },
    )


def add_task_log(task_id: str, level: str, message: str):
    """Add a log entry for a task (compatible with existing API)"""
    key = f"task:{task_id}:logs"
    log_entry = {"ts": time.time(), "level": level, "message": message}
    redis_client.rpush(key, json.dumps(log_entry))


def get_task_logs(task_id: str, tail: int = 100) -> List[Dict[str, Any]]:
    """Get task logs (compatible with existing API)"""
    key = f"task:{task_id}:logs"
    items = redis_client.lrange(key, -tail, -1)
    return [json.loads(item) for item in items]


def add_task_artifact(task_id: str, artifact: Dict[str, Any]):
    """Add an artifact for a task (compatible with existing API)"""
    key = f"task:{task_id}:artifacts"
    redis_client.rpush(key, json.dumps(artifact))


def get_task_artifacts(task_id: str) -> List[Dict[str, Any]]:
    """Get task artifacts (compatible with existing API)"""
    key = f"task:{task_id}:artifacts"
    items = redis_client.lrange(key, 0, -1)
    return [json.loads(item) for item in items]


def get_task_meta(task_id: str) -> Dict[str, Any]:
    """Get task metadata (compatible with existing API)"""
    key = f"task:{task_id}"
    raw = redis_client.hgetall(key)
    if not raw:
        return {}

    # Decode bytes to strings
    decoded = {}
    for k, v in raw.items():
        k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
        v_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        decoded[k_str] = v_str

    # Try to parse payload if it exists
    if "payload" in decoded:
        try:
            decoded["payload"] = json.loads(decoded["payload"])
        except (json.JSONDecodeError, TypeError):
            pass  # Keep as string if parsing fails

    return decoded


def get_celery_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get Celery task status and result if available.
    This provides additional Celery-specific information.
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        return {
            "celery_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "date_done": result.date_done.isoformat() if result.date_done else None,
            "traceback": result.traceback if result.failed() else None,
        }
    except Exception as e:
        logger.warning(f"Failed to get Celery status for task {task_id}: {e}")
        return None


def clear_task(task_id: str):
    """Clear task data from Redis (compatible with existing API)"""
    redis_client.delete(f"task:{task_id}")
    redis_client.delete(f"task:{task_id}:logs")
    redis_client.delete(f"task:{task_id}:artifacts")


def get_all_tasks() -> List[Dict[str, Any]]:
    """Get all tasks from Redis (for monitoring/debugging)"""
    keys = redis_client.keys("task:*")
    tasks = []

    for key in keys:
        key_str = key.decode("utf-8") if isinstance(key, bytes) else key
        if ":logs" in key_str or ":artifacts" in key_str:
            continue

        task_id = key_str.split(":", 1)[1]
        meta = get_task_meta(task_id)
        celery_status = get_celery_task_status(task_id)

        if meta:
            task_info = {"task_id": task_id, **meta, "celery_status": celery_status}
            tasks.append(task_info)

    return tasks


# Legacy alias for backward compatibility
r = redis_client
