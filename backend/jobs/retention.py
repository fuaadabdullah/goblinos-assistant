"""
Data-retention purge job for APScheduler.

Deletes user-content rows (tasks, routing_requests, search_documents) older
than a configurable retention window. Runs daily with Redis locking to
prevent multiple replicas from executing simultaneously.

Retention window: DATA_RETENTION_DAYS env var (default 90 days).
See docs/privacy.md for what is stored and for how long.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from sqlalchemy import text

import database

from scheduler import with_redis_lock

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 90

# (table, timestamp column) pairs holding user content subject to retention.
RETENTION_TABLES = (
    ("tasks", "created_at"),
    ("routing_requests", "created_at"),
    ("search_documents", "created_at"),
)


def get_retention_days() -> int:
    """Return the configured retention window in days (default 90)."""
    try:
        return max(1, int(os.getenv("DATA_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS))))
    except (TypeError, ValueError):
        return DEFAULT_RETENTION_DAYS


def purge_old_records(older_than_days: Optional[int] = None) -> Dict[str, int]:
    """
    Delete rows older than the retention window from user-content tables.

    Args:
        older_than_days: Retention window in days. Defaults to the
            DATA_RETENTION_DAYS env var (90 if unset/invalid).

    Returns:
        Dict mapping table name -> number of rows deleted.
    """
    if older_than_days is None:
        older_than_days = get_retention_days()
    older_than_days = max(1, int(older_than_days))

    cutoff = datetime.utcnow() - timedelta(days=older_than_days)
    deleted: Dict[str, int] = {}

    db = database.SessionLocal()
    try:
        for table, ts_column in RETENTION_TABLES:
            try:
                result = db.execute(
                    text(f"DELETE FROM {table} WHERE {ts_column} < :cutoff"),
                    {"cutoff": cutoff},
                )
                count = result.rowcount
                try:
                    count = int(count)
                except Exception:
                    count = 0
                # Commit per table so one table's failure cannot roll back
                # another table's already-completed purge.
                db.commit()
                deleted[table] = count
                logger.info(
                    "Retention purge: deleted %d rows older than %d days from %s",
                    count,
                    older_than_days,
                    table,
                )
            except Exception as exc:
                # Best-effort per table: schema drift across deployments means
                # a table may be missing; never let one table abort the purge.
                db.rollback()
                deleted[table] = 0
                logger.warning(
                    "Retention purge skipped table %s: %s", table, exc
                )
    finally:
        db.close()

    return deleted


@with_redis_lock("retention_purge", ttl=86400)  # 24 hour TTL
def retention_purge_job():
    """
    APScheduler job to purge user-content rows past the retention window.

    Runs daily. Uses the DATA_RETENTION_DAYS env var (default 90 days).
    """
    logger.info(
        "Starting retention purge job (retention_days=%d)", get_retention_days()
    )

    try:
        start_time = time.time()
        deleted = purge_old_records()
        total = sum(deleted.values())
        duration = time.time() - start_time
        logger.info(
            "Retention purge job completed in %.2fs: deleted %d records (%s)",
            duration,
            total,
            deleted,
        )
    except Exception as e:
        logger.error("Retention purge job failed: %s", e)
        raise
