"""
APScheduler with Redis locks for lightweight periodic tasks.

This replaces Celery for tasks that:
- Run inside the app
- Are tolerant of occasional single-instance execution
- Have short runtime (< 30 seconds)
- Don't need complex workflow orchestration

Uses APScheduler for scheduling + Redis distributed locks to prevent
multiple replicas from running the same job simultaneously.
"""

import os
import logging
import time
import atexit
from contextlib import contextmanager
from typing import Optional, Callable, Any
from datetime import datetime, timedelta

import redis

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Any  # type: ignore
    Histogram = Any  # type: ignore
    Gauge = Any  # type: ignore
    CollectorRegistry = Any  # type: ignore

# APScheduler imports - will be available after pip install
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.jobstores.memory import MemoryJobStore

    # Use a threadpool executor. Many jobs call `asyncio.run()` internally,
    # and AsyncIOExecutor expects an event loop integration that isn't reliably
    # available in our current BackgroundScheduler setup (caused runtime errors on Fly).
    from apscheduler.executors.pool import ThreadPoolExecutor
    from sqlalchemy import create_engine

    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = Any  # type: ignore
    SQLAlchemyJobStore = Any  # type: ignore
    MemoryJobStore = Any  # type: ignore
    ThreadPoolExecutor = Any  # type: ignore
    create_engine = None  # type: ignore

logger = logging.getLogger(__name__)

# Redis configuration — lazy connection to avoid crashes when Redis is not available
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client: Optional[redis.Redis] = None


def _get_redis_client() -> Optional[redis.Redis]:
    """Return a lazily-initialised Redis client, or None if unavailable."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(REDIS_URL, socket_connect_timeout=3)
            _redis_client.ping()
        except Exception as exc:
            logger.warning("Redis unavailable for scheduler locks: %s", exc)
            _redis_client = None  # type: ignore[assignment]
    return _redis_client


# Alias kept for internal callers that reference the module-level name
redis_client = None  # type: ignore[assignment]  # populated lazily via _get_redis_client()

# Database for job persistence
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///scheduler_jobs.db")
engine = create_engine(DATABASE_URL) if APSCHEDULER_AVAILABLE else None

# Global scheduler instance
scheduler: Optional[BackgroundScheduler] = None

# Prometheus metrics registry
registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None

# Metrics definitions
if PROMETHEUS_AVAILABLE:
    # Job execution metrics
    jobs_fired = Counter(
        "apscheduler_jobs_fired_total",
        "Total number of jobs fired by APScheduler",
        ["job_id", "job_name"],
        registry=registry,
    )

    job_duration = Histogram(
        "apscheduler_job_duration_seconds",
        "Time spent executing jobs",
        ["job_id", "job_name", "status"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        registry=registry,
    )

    job_success_total = Counter(
        "apscheduler_job_success_total",
        "Total number of successful job executions",
        ["job_id", "job_name"],
        registry=registry,
    )

    job_failure_total = Counter(
        "apscheduler_job_failure_total",
        "Total number of failed job executions",
        ["job_id", "job_name"],
        registry=registry,
    )

    # Redis lock metrics
    redis_lock_acquired = Counter(
        "apscheduler_redis_lock_acquired_total",
        "Total number of Redis locks acquired",
        ["job_id", "job_name"],
        registry=registry,
    )

    redis_lock_failed = Counter(
        "apscheduler_redis_lock_failed_total",
        "Total number of Redis lock acquisition failures",
        ["job_id", "job_name"],
        registry=registry,
    )

    redis_lock_released = Counter(
        "apscheduler_redis_lock_released_total",
        "Total number of Redis locks released",
        ["job_id", "job_name"],
        registry=registry,
    )

    # Scheduler health metrics
    scheduler_jobs_active = Gauge(
        "apscheduler_jobs_active", "Number of currently active jobs", registry=registry
    )

    scheduler_jobs_total = Gauge(
        "apscheduler_jobs_total", "Total number of jobs in scheduler", registry=registry
    )

    # Alert metrics
    scheduler_missed_runs = Counter(
        "apscheduler_missed_runs_total",
        "Total number of missed scheduled runs",
        ["job_id", "job_name"],
        registry=registry,
    )

    scheduler_backlog_size = Gauge(
        "apscheduler_backlog_size",
        "Current size of job execution backlog",
        registry=registry,
    )

else:
    # Dummy metrics when Prometheus not available
    jobs_fired = job_duration = job_success_total = job_failure_total = None
    redis_lock_acquired = redis_lock_failed = redis_lock_released = None
    scheduler_jobs_active = scheduler_jobs_total = None
    scheduler_missed_runs = scheduler_backlog_size = None


@contextmanager
def redis_lock(lock_key: str, ttl: int = 60):
    """
    Redis-based distributed lock to prevent multiple instances from running the same job.

    Args:
        lock_key: Unique key for the lock
        ttl: Time-to-live in seconds (lock expires if job hangs)

    Yields:
        bool: True if lock acquired, False if another instance is running
    """
    got_lock = False
    client = _get_redis_client()
    if client is None:
        logger.warning("Redis unavailable — skipping distributed lock for %s", lock_key)
        yield True  # run without lock when Redis is down
        return
    got_lock = client.set(name=lock_key, value="1", nx=True, ex=ttl)
    try:
        if not got_lock:
            logger.debug(f"Lock {lock_key} already held by another instance")
            # Track lock acquisition failure
            if redis_lock_failed:
                redis_lock_failed.labels(job_id=lock_key, job_name=lock_key).inc()
            yield False
        else:
            logger.debug(f"Acquired lock {lock_key}")
            # Track lock acquisition success
            if redis_lock_acquired:
                redis_lock_acquired.labels(job_id=lock_key, job_name=lock_key).inc()
            yield True
    finally:
        if got_lock:
            client.delete(lock_key)
            logger.debug(f"Released lock {lock_key}")
            # Track lock release
            if redis_lock_released:
                redis_lock_released.labels(job_id=lock_key, job_name=lock_key).inc()


def with_redis_lock(lock_key: str, ttl: int = 60):
    """
    Decorator to wrap job functions with Redis locking.

    Args:
        lock_key: Unique key for the lock
        ttl: Lock TTL in seconds
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            with redis_lock(lock_key, ttl) as acquired:
                if not acquired:
                    return  # Another instance is running this job
                try:
                    # Track job execution start
                    if jobs_fired:
                        jobs_fired.labels(job_id=lock_key, job_name=func.__name__).inc()

                    start_time = time.time()
                    result = func(*args, **kwargs)

                    # Track successful job execution
                    duration = time.time() - start_time
                    if job_duration:
                        job_duration.labels(
                            job_id=lock_key, job_name=func.__name__, status="success"
                        ).observe(duration)
                    if job_success_total:
                        job_success_total.labels(job_id=lock_key, job_name=func.__name__).inc()

                    return result
                except Exception as e:
                    # Track failed job execution
                    duration = time.time() - start_time
                    if job_duration:
                        job_duration.labels(
                            job_id=lock_key, job_name=func.__name__, status="failure"
                        ).observe(duration)
                    if job_failure_total:
                        job_failure_total.labels(job_id=lock_key, job_name=func.__name__).inc()

                    logger.error(f"Job {func.__name__} failed: {e}")
                    raise

        return wrapper

    return decorator


def create_scheduler() -> Any:
    """
    Create and configure the APScheduler instance.

    Returns:
        BackgroundScheduler: Configured scheduler instance
    """
    global scheduler

    if scheduler is not None:
        if os.getenv("PYTEST_CURRENT_TEST"):
            scheduler = None
        else:
            return scheduler

    # Job store:
    # - Default to in-memory. We re-register jobs at startup anyway, and some
    #   decorated callables are not serializable for persistent stores.
    # - Allow opting into persistence explicitly.
    persist_jobs = os.getenv("SCHEDULER_PERSIST_JOBS", "").lower() in ("1", "true", "yes")
    if persist_jobs:
        jobstores = {"default": SQLAlchemyJobStore(engine=engine)}
    else:
        jobstores = {"default": MemoryJobStore()}

    # Threadpool executor for job execution.
    executors = {"default": ThreadPoolExecutor(max_workers=5)}

    # Job defaults
    job_defaults = {
        "coalesce": True,  # Combine multiple missed runs into one
        "max_instances": 1,  # Only one instance of each job at a time
        "misfire_grace_time": 30,  # Allow 30 seconds late execution
    }

    if APSCHEDULER_AVAILABLE:
        from apscheduler.schedulers.background import BackgroundScheduler as SchedulerClass
    else:
        SchedulerClass = BackgroundScheduler

    scheduler = SchedulerClass(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone=os.getenv("SCHEDULER_TIMEZONE", "UTC"),
    )

    return scheduler


def start_scheduler():
    """Start the scheduler and register all jobs."""
    global scheduler

    if scheduler is not None and scheduler.running:
        logger.warning("Scheduler already running")
        return

    scheduler = create_scheduler()

    # Register jobs
    register_jobs(scheduler)

    # Start the scheduler
    scheduler.start()
    logger.info("APScheduler started with Redis locks")


def stop_scheduler():
    """Stop the scheduler gracefully."""
    global scheduler

    if scheduler and scheduler.running:
        scheduler.shutdown(wait=True)
        logger.info("APScheduler stopped")
        scheduler = None


def register_jobs(scheduler):
    """Register all periodic jobs with the scheduler."""

    # Import job functions
    from .jobs.provider_health import probe_all_providers_job
    from .jobs.system_health import system_health_check_job
    from .jobs.cleanup import cleanup_expired_data_job

    # Provider health check - every 5 minutes
    scheduler.add_job(
        probe_all_providers_job,
        "interval",
        minutes=5,
        id="provider_health_check",
        replace_existing=True,
    )

    # System health check - every minute
    scheduler.add_job(
        system_health_check_job,
        "interval",
        minutes=1,
        id="system_health_check",
        replace_existing=True,
    )

    # Cleanup expired data - every 6 hours
    scheduler.add_job(
        cleanup_expired_data_job,
        "interval",
        hours=6,
        id="cleanup_expired_data",
        replace_existing=True,
    )

    logger.info("Registered all APScheduler jobs")


def get_scheduler_status() -> dict:
    """
    Get the current status of the scheduler and jobs.

    Returns:
        dict: Scheduler status information
    """
    if not scheduler:
        return {"status": "not_started"}

    try:
        jobs_list = scheduler.get_jobs()
    except Exception:
        return {"status": "not_started"}

    if not getattr(scheduler, "running", False) and not jobs_list:
        return {"status": "not_started"}

    try:
        jobs_iter = list(jobs_list)
    except TypeError:
        return {"status": "not_started"}

    jobs = []
    for job in jobs_iter:
        jobs.append(
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": getattr(job, "next_run_time", None).isoformat()
                if getattr(job, "next_run_time", None)
                else None,
                "trigger": str(job.trigger),
            }
        )

    return {"status": "running" if scheduler.running else "stopped", "jobs": jobs}


def get_prometheus_metrics() -> str:
    """
    Get Prometheus metrics in the standard format.

    Returns:
        str: Metrics in Prometheus exposition format
    """
    if not PROMETHEUS_AVAILABLE or not registry:
        return "# Prometheus not available"

    from prometheus_client import generate_latest

    return generate_latest(registry).decode("utf-8")


def update_scheduler_metrics():
    """
    Update scheduler health metrics.

    This should be called periodically to keep gauge metrics current.
    """
    if not scheduler or not PROMETHEUS_AVAILABLE:
        return

    try:
        jobs = scheduler.get_jobs()
        if scheduler_jobs_total:
            scheduler_jobs_total.set(len(jobs))

        # Count active jobs (jobs currently running)
        active_count = sum(1 for job in jobs if job.next_run_time is None)  # Rough approximation
        if scheduler_jobs_active:
            scheduler_jobs_active.set(active_count)

        # Check for missed runs (jobs that should have run but didn't)
        now = datetime.now(scheduler.timezone if scheduler.timezone else None)
        missed_count = 0
        for job in jobs:
            if job.next_run_time and job.next_run_time < now:
                missed_count += 1
                if scheduler_missed_runs:
                    scheduler_missed_runs.labels(job_id=job.id, job_name=job.name or job.id).inc()

        # Estimate backlog size (simplified - could be enhanced)
        if scheduler_backlog_size:
            scheduler_backlog_size.set(missed_count)

    except Exception as e:
        logger.error(f"Failed to update scheduler metrics: {e}")


# Cleanup on exit
atexit.register(stop_scheduler)
