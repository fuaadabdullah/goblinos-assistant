import os
import redis
import json
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_redis_client: Optional[redis.Redis] = None


def _get_redis() -> redis.Redis:
    """Lazy Redis connection — connects on first use, not at import time."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            REDIS_URL,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        # Verify the connection is alive
        _redis_client.ping()
        logger.info(
            "Redis connected: %s", REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL
        )
    return _redis_client


def enqueue_task(task_id: str, payload: Dict[str, Any]) -> str:
    r = _get_redis()
    key = f"task:{task_id}"
    r.hset(
        key, mapping={"status": "queued", "created_at": time.time(), "payload": json.dumps(payload)}
    )
    r.lpush("tasks:queue", task_id)
    return task_id


def set_task_running(task_id: str):
    r = _get_redis()
    key = f"task:{task_id}"
    r.hset(key, mapping={"status": "running", "started_at": time.time()})


def set_task_completed(task_id: str, result: str):
    r = _get_redis()
    key = f"task:{task_id}"
    r.hset(key, mapping={"status": "completed", "result": result, "completed_at": time.time()})


def add_task_log(task_id: str, level: str, message: str):
    r = _get_redis()
    key = f"task:{task_id}:logs"
    r.rpush(key, json.dumps({"ts": time.time(), "level": level, "message": message}))


def get_task_logs(task_id: str, tail: int = 100):
    r = _get_redis()
    key = f"task:{task_id}:logs"
    items = r.lrange(key, -tail, -1)
    return [json.loads(i) for i in items]


def add_task_artifact(task_id: str, artifact: Dict[str, Any]):
    r = _get_redis()
    key = f"task:{task_id}:artifacts"
    r.rpush(key, json.dumps(artifact))


def get_task_artifacts(task_id: str):
    r = _get_redis()
    key = f"task:{task_id}:artifacts"
    items = r.lrange(key, 0, -1)
    return [json.loads(i) for i in items]


def get_task_meta(task_id: str) -> Dict[str, Any]:
    r = _get_redis()
    key = f"task:{task_id}"
    raw = r.hgetall(key)
    if not raw:
        return {}
    decoded = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw.items()}
    return decoded


def clear_task(task_id: str):
    r = _get_redis()
    key = f"task:{task_id}"
    r.delete(key)
    r.delete(f"task:{task_id}:logs")
    r.delete(f"task:{task_id}:artifacts")


def process_task(task_id: str):
    """Process a task in a worker process; used by RQ to execute jobs."""
    meta = get_task_meta(task_id)
    if not meta:
        add_task_log(task_id, "error", "Task metadata not found")
        return

    set_task_running(task_id)
    add_task_log(task_id, "info", "Task started")
    time.sleep(2)
    artifact = {"name": "result.txt", "contents": f"Task {task_id} executed"}
    add_task_artifact(task_id, artifact)
    add_task_log(task_id, "info", "Task completed and artifact created")
    set_task_completed(task_id, result="success")
