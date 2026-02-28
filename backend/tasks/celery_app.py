"""
Celery Application Configuration
Complex task management for distributed AI workloads
"""

from celery import Celery

from ..config import settings

# Create Celery app
app = Celery(
    "goblin",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "src.tasks.inference_tasks",
        "src.tasks.training_tasks",
        "src.tasks.batch_tasks",
    ],
)

# Configuration
app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_acks_late=True,  # Ack after completion for reliability
    task_reject_on_worker_lost=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3300,  # 55 min soft limit
    
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for GPU workloads
    worker_concurrency=1,  # Single worker for GPU tasks
    
    # Result settings
    result_expires=86400,  # 24 hours
    
    # Task routing
    task_routes={
        # High priority: RunPod production inference
        "src.tasks.inference_tasks.inference_high_priority": {
            "queue": "high_priority"
        },
        # Batch: Vast.ai cost-optimized
        "src.tasks.batch_tasks.*": {
            "queue": "batch"
        },
        # Training: RunPod or Vast.ai depending on config
        "src.tasks.training_tasks.*": {
            "queue": "training"
        },
        # Default: GCP development
        "src.tasks.inference_tasks.inference_default": {
            "queue": "default"
        },
    },
    
    # Queue defaults
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "health-check-every-5-minutes": {
            "task": "src.tasks.batch_tasks.provider_health_check",
            "schedule": 300.0,  # 5 minutes
        },
        "cost-report-daily": {
            "task": "src.tasks.batch_tasks.daily_cost_report",
            "schedule": 86400.0,  # 24 hours
        },
        "checkpoint-cleanup-weekly": {
            "task": "src.tasks.batch_tasks.cleanup_old_checkpoints",
            "schedule": 604800.0,  # 7 days
        },
    },
)


def main() -> None:
    """Entry point for Celery worker."""
    app.start()


if __name__ == "__main__":
    main()
