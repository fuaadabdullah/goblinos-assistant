"""
Monitoring and Observability Configuration for Goblin Assistant Celery

Integrates with Flower for Celery monitoring and provides health checks.
"""

import os
from flask import Flask, jsonify
from .celery_app import app as celery_app
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Flask app for health checks and monitoring endpoints
monitoring_app = Flask(__name__)


@monitoring_app.route("/health")
def health_check():
    """Basic health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "goblin-assistant-celery",
        }
    )


@monitoring_app.route("/health/celery")
def celery_health_check():
    """Detailed Celery health check."""
    try:
        # Check if Celery is responsive
        inspect = celery_app.control.inspect()

        # Get basic stats
        stats = inspect.stats()
        active = inspect.active()
        scheduled = inspect.scheduled()

        # Calculate some metrics
        total_workers = len(stats or {})
        total_active_tasks = sum(len(tasks) for tasks in (active or {}).values())
        total_scheduled_tasks = sum(len(tasks) for tasks in (scheduled or {}).values())

        return jsonify(
            {
                "status": "healthy",
                "celery_status": "running",
                "workers": total_workers,
                "active_tasks": total_active_tasks,
                "scheduled_tasks": total_scheduled_tasks,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Celery health check failed: {e}")
        return jsonify(
            {
                "status": "unhealthy",
                "celery_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
        ), 503


@monitoring_app.route("/metrics/celery")
def celery_metrics():
    """Expose Celery metrics for monitoring systems."""
    try:
        inspect = celery_app.control.inspect()

        metrics = {
            "celery_workers_total": len(inspect.stats() or {}),
            "celery_tasks_active": sum(
                len(tasks) for tasks in (inspect.active() or {}).values()
            ),
            "celery_tasks_scheduled": sum(
                len(tasks) for tasks in (inspect.scheduled() or {}).values()
            ),
            "celery_tasks_reserved": sum(
                len(tasks) for tasks in (inspect.reserved() or {}).values()
            ),
        }

        # Add per-worker metrics
        if inspect.stats():
            for worker, stats in inspect.stats().items():
                worker_name = worker.replace("@", "_").replace(".", "_")
                metrics[f"celery_worker_{worker_name}_active"] = stats.get("active", 0)
                metrics[f"celery_worker_{worker_name}_processed"] = stats.get(
                    "total", {}
                ).get("tasks", 0)

        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Failed to collect Celery metrics: {e}")
        return jsonify({"error": str(e)}), 500


@monitoring_app.route("/jobs/<job_id>")
def get_job_details(job_id: str):
    """Get detailed information about a specific job."""
    try:
        from .job_workflows import JobMonitor

        job_info = JobMonitor.get_job_status(job_id)
        return jsonify(job_info)
    except Exception as e:
        logger.error(f"Failed to get job details for {job_id}: {e}")
        return jsonify({"error": str(e)}), 500


@monitoring_app.route("/jobs/active")
def get_active_jobs():
    """Get all currently active jobs."""
    try:
        from .job_workflows import JobMonitor

        active_jobs = JobMonitor.get_active_jobs()
        return jsonify({"active_jobs": active_jobs})
    except Exception as e:
        logger.error(f"Failed to get active jobs: {e}")
        return jsonify({"error": str(e)}), 500


@monitoring_app.route("/jobs/stats")
def get_job_stats():
    """Get comprehensive job queue statistics."""
    try:
        from .job_workflows import JobMonitor

        stats = JobMonitor.get_job_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Failed to get job stats: {e}")
        return jsonify({"error": str(e)}), 500


@monitoring_app.route("/jobs/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id: str):
    """Cancel a running job."""
    try:
        from .job_workflows import JobMonitor

        success = JobMonitor.cancel_job(job_id)
        return jsonify({"success": success, "job_id": job_id})
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        return jsonify({"error": str(e)}), 500


def create_flower_config():
    """Create Flower configuration for Celery monitoring."""
    return {
        "broker_api": os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        "url_prefix": "/flower",
        "basic_auth": os.getenv(
            "FLOWER_BASIC_AUTH"
        ),  # Format: user1:password1,user2:password2
        "max_tasks": 10000,
        "db": "flower.db",  # SQLite database for Flower
        "persistent": True,
        "state_save_interval": 0,  # Save state immediately
        "auto_refresh": True,
        "inspect_timeout": 1000,
        "enable_events": True,
    }


def start_monitoring_server(host: str = "0.0.0.0", port: int = 5555):
    """Start the monitoring Flask server."""
    logger.info(f"Starting monitoring server on {host}:{port}")
    monitoring_app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    start_monitoring_server()
