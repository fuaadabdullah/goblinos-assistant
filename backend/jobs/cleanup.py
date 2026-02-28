"""
Cleanup expired data job for APScheduler.

Replaces the Celery task: tasks.cleanup_worker.cleanup_expired_data
Runs every 6 hours with Redis locking to prevent multiple instances.
"""

import logging
import time
from datetime import datetime, timedelta
from sqlalchemy import text

import database

from scheduler import with_redis_lock

logger = logging.getLogger(__name__)


@with_redis_lock("cleanup_expired_data", ttl=21600)  # 6 hour TTL
def cleanup_expired_data_job():
    """
    APScheduler job to cleanup expired data from the database.

    This replaces the Celery scheduled task and runs with Redis locking
    to prevent multiple replicas from executing simultaneously.

    Cleans up:
    - Old inference logs (older than 30 days)
    - Expired sessions (older than 24 hours)
    - Old metrics data (older than 90 days)
    - Temporary files and artifacts
    """
    logger.info("Starting cleanup expired data job")

    try:
        start_time = time.time()

        # Perform cleanup operations
        cleanup_results = _perform_cleanup()

        # Log results
        duration = time.time() - start_time
        logger.info(
            f"Cleanup job completed in {duration:.2f}s: "
            f"deleted {cleanup_results['total_deleted']} records, "
            f"freed {cleanup_results['space_freed_mb']:.2f} MB"
        )

    except Exception as e:
        logger.error(f"Cleanup expired data job failed: {e}")
        raise


def _perform_cleanup() -> dict:
    """
    Perform actual cleanup operations.

    Returns:
        Dict with cleanup statistics
    """
    db = database.SessionLocal()
    total_deleted = 0
    space_freed_mb = 0.0

    try:
        # Cleanup old inference logs (older than 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        result = db.execute(
            text("DELETE FROM inference_logs WHERE created_at < :cutoff"),
            {"cutoff": thirty_days_ago},
        )
        logs_deleted = result.rowcount
        try:
            logs_deleted = int(logs_deleted)
        except Exception:
            logs_deleted = 0
        total_deleted += logs_deleted
        logger.debug(f"Deleted {logs_deleted} old inference logs")

        # Cleanup expired sessions (older than 24 hours)
        result = db.execute(
            text("DELETE FROM user_sessions WHERE expires_at < :now"),
            {"now": datetime.utcnow()},
        )
        sessions_deleted = result.rowcount
        try:
            sessions_deleted = int(sessions_deleted)
        except Exception:
            sessions_deleted = 0
        total_deleted += sessions_deleted
        logger.debug(f"Deleted {sessions_deleted} expired sessions")

        # Cleanup old provider metrics (older than 90 days)
        ninety_days_ago = datetime.utcnow() - timedelta(days=90)
        result = db.execute(
            text("DELETE FROM provider_metrics WHERE timestamp < :cutoff"),
            {"cutoff": ninety_days_ago},
        )
        metrics_deleted = result.rowcount
        try:
            metrics_deleted = int(metrics_deleted)
        except Exception:
            metrics_deleted = 0
        total_deleted += metrics_deleted
        logger.debug(f"Deleted {metrics_deleted} old provider metrics")

        # Cleanup old chat history (older than 7 days for anonymous users)
        # This is a simplified example - adjust based on your retention policy
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        result = db.execute(
            text("""
                DELETE FROM chat_messages
                WHERE created_at < :cutoff
                AND user_id IS NULL
            """),
            {"cutoff": seven_days_ago},
        )
        messages_deleted = result.rowcount
        try:
            messages_deleted = int(messages_deleted)
        except Exception:
            messages_deleted = 0
        total_deleted += messages_deleted
        logger.debug(f"Deleted {messages_deleted} old anonymous chat messages")

        # Estimate space freed (rough calculation)
        # This is approximate - in production you'd track actual table sizes
        space_freed_mb = (
            logs_deleted * 0.5  # ~0.5KB per log
            + sessions_deleted * 0.1  # ~0.1KB per session
            + metrics_deleted * 0.2  # ~0.2KB per metric
            + messages_deleted * 1.0  # ~1KB per message
        ) / 1024  # Convert to MB

        db.commit()

        return {
            "total_deleted": total_deleted,
            "space_freed_mb": space_freed_mb,
            "breakdown": {
                "inference_logs": logs_deleted,
                "user_sessions": sessions_deleted,
                "provider_metrics": metrics_deleted,
                "chat_messages": messages_deleted,
            },
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error during cleanup: {e}")
        raise
    finally:
        db.close()


def get_cleanup_stats() -> dict:
    """
    Get statistics about data that could be cleaned up.

    Useful for monitoring and deciding whether to adjust cleanup schedules.

    Returns:
        Dict with cleanup statistics
    """
    db = database.SessionLocal()

    try:
        # Count old records that would be deleted
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        logs_count = db.execute(
            text("SELECT COUNT(*) FROM inference_logs WHERE created_at < :cutoff"),
            {"cutoff": thirty_days_ago},
        ).scalar()

        sessions_count = db.execute(
            text("SELECT COUNT(*) FROM user_sessions WHERE expires_at < :now"),
            {"now": datetime.utcnow()},
        ).scalar()

        ninety_days_ago = datetime.utcnow() - timedelta(days=90)
        metrics_count = db.execute(
            text("SELECT COUNT(*) FROM provider_metrics WHERE timestamp < :cutoff"),
            {"cutoff": ninety_days_ago},
        ).scalar()

        return {
            "pending_cleanup": {
                "inference_logs_30d": logs_count or 0,
                "expired_sessions": sessions_count or 0,
                "provider_metrics_90d": metrics_count or 0,
            },
            "total_pending": (logs_count or 0) + (sessions_count or 0) + (metrics_count or 0),
        }

    except Exception as e:
        logger.error(f"Failed to get cleanup stats: {e}")
        return {"error": str(e)}
    finally:
        db.close()
