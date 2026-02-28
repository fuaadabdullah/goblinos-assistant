"""
Celery Application Configuration for Goblin Assistant

Provides enterprise-grade job queue management with:
- Priority queues
- Delayed jobs with precision
- Job dependencies (chains, groups, chords)
- Comprehensive error handling and retry logic
- Monitoring and observability
- Result storage and tracking
"""

import os
import logging
from celery import Celery, Task
from kombu import Exchange, Queue

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Celery app configuration
app = Celery(
    "goblin_assistant",
    broker=REDIS_URL,
    backend=RESULT_BACKEND,
    include=[
        "tasks.provider_probe_worker",
        "tasks.model_training_worker",
    ],
)

# Timezone
app.conf.timezone = os.getenv("CELERY_TIMEZONE", "UTC")

# Task serialization
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]

# Result storage
app.conf.result_expires = 3600  # 1 hour
app.conf.result_cache_max = 10000

# Worker configuration
app.conf.worker_prefetch_multiplier = 1
app.conf.worker_max_tasks_per_child = 1000
app.conf.worker_disable_rate_limits = False

# Task routing and queues with priorities
app.conf.task_queues = (
    Queue("high_priority", Exchange("high_priority"), routing_key="high_priority"),
    Queue("default", Exchange("default"), routing_key="default"),
    Queue("low_priority", Exchange("low_priority"), routing_key="low_priority"),
    Queue("scheduled", Exchange("scheduled"), routing_key="scheduled"),
    Queue("batch", Exchange("batch"), routing_key="batch"),
    # Cloud infrastructure queues
    Queue("training", Exchange("training"), routing_key="training"),
    Queue("inference", Exchange("inference"), routing_key="inference"),
)

app.conf.task_default_queue = "default"
app.conf.task_default_exchange = "default"
app.conf.task_default_routing_key = "default"

# Route tasks to appropriate queues
app.conf.task_routes = {
    "tasks.provider_probe_worker.*": {"queue": "high_priority"},
    "tasks.model_training_worker.train_model_task": {"queue": "training"},
    "tasks.model_training_worker.fine_tune_model_task": {"queue": "training"},
    "tasks.model_training_worker.distributed_training_task": {"queue": "training"},
    "tasks.model_training_worker.monitor_training_job_task": {"queue": "training"},
    "tasks.model_training_worker.sync_models_task": {"queue": "default"},
    "tasks.model_training_worker.cleanup_old_models_task": {"queue": "low_priority"},
}

# Task execution settings
app.conf.task_acks_late = True
app.conf.task_reject_on_worker_lost = True
app.conf.task_send_sent_event = True

# Error handling and retries
app.conf.task_default_retry_delay = 60  # 1 minute
app.conf.task_max_retries = 3
app.conf.task_retry_backoff = True
app.conf.task_retry_backoff_max = 700  # Max 12 minutes
app.conf.task_retry_jitter = True

# Circuit breaker settings
app.conf.worker_disable_rate_limits = False

# Monitoring and logging
app.conf.worker_send_task_events = True
app.conf.task_send_sent_event = True
app.conf.task_ignore_result = False

# Scheduled tasks (beat configuration)
app.conf.beat_schedule = {
    # TODO: Add actual heavy tasks here when implemented
    # Currently no active Celery beat schedules - using APScheduler for lightweight tasks
}

# Custom task classes for advanced features
logger = logging.getLogger(__name__)


class GoblinTask(Task):
    """Custom task class with enhanced error handling and monitoring."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure with enhanced logging."""
        logger.error(
            f"Task {self.name}[{task_id}] failed: {exc}",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "args": args,
                "kwargs": kwargs,
                "exception": str(exc),
                "traceback": str(einfo),
            },
        )
        # Send alert to monitoring system
        self._send_failure_alert(task_id, exc, einfo)

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success with logging."""
        logger.info(
            f"Task {self.name}[{task_id}] completed successfully",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "args": args,
                "kwargs": kwargs,
                "result": str(retval)[:500],  # Truncate long results
            },
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(
            f"Task {self.name}[{task_id}] retrying: {exc}",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "retry_count": self.request.retries,
                "max_retries": self.max_retries,
            },
        )

    def _send_failure_alert(self, task_id: str, exc: Exception, einfo):
        """Send failure alert to monitoring system."""
        try:
            # Integration point for alerting system (Sentry, PagerDuty, etc.)
            from monitoring import send_alert

            send_alert(
                title=f"Celery Task Failed: {self.name}",
                message=f"Task {task_id} failed after {self.request.retries} retries",
                level="error",
                extra={
                    "task_id": task_id,
                    "task_name": self.name,
                    "exception": str(exc),
                    "traceback": str(einfo),
                },
            )
        except Exception as alert_error:
            logger.error(f"Failed to send failure alert: {alert_error}")


# Register custom task class
app.conf.task_cls = "celery_app.GoblinTask"

# Import tasks to register them
if __name__ == "__main__":
    app.start()
