"""
System health check job for APScheduler.

Replaces the Celery task: tasks.monitoring_worker.system_health_check
Runs every minute with Redis locking to prevent multiple instances.
"""

import logging
import psutil
import time
from typing import Dict, Any

from scheduler import with_redis_lock

logger = logging.getLogger(__name__)


@with_redis_lock("system_health_check", ttl=60)
def system_health_check_job():
    """
    APScheduler job to perform system health checks.

    This replaces the Celery scheduled task and runs with Redis locking
    to prevent multiple replicas from executing simultaneously.

    Checks:
    - CPU usage
    - Memory usage
    - Disk usage
    - Basic service availability
    """
    logger.debug("Starting system health check job")

    try:
        start_time = time.time()

        # Perform health checks
        health_data = _perform_health_checks()

        # Log results
        duration = time.time() - start_time
        logger.info(
            f"System health check completed in {duration:.2f}s: "
            f"CPU={health_data['cpu_percent']}%, "
            f"Memory={health_data['memory_percent']}%, "
            f"Disk={health_data['disk_percent']}%"
        )

        # In production, you might want to:
        # - Send metrics to monitoring system (Datadog, etc.)
        # - Alert if thresholds are exceeded
        # - Store historical data

        # Check for critical issues
        if health_data["cpu_percent"] > 90:
            logger.warning(f"High CPU usage: {health_data['cpu_percent']}%")
        if health_data["memory_percent"] > 90:
            logger.warning(f"High memory usage: {health_data['memory_percent']}%")
        if health_data["disk_percent"] > 90:
            logger.warning(f"High disk usage: {health_data['disk_percent']}%")

    except Exception as e:
        logger.error(f"System health check job failed: {e}")
        raise


def _perform_health_checks() -> Dict[str, Any]:
    """
    Perform actual system health checks.

    Returns:
        Dict with health metrics
    """
    health_data = {}

    try:
        # CPU usage (average over 1 second)
        health_data["cpu_percent"] = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        health_data["memory_percent"] = memory.percent
        health_data["memory_used_gb"] = memory.used / (1024**3)
        health_data["memory_total_gb"] = memory.total / (1024**3)

        # Disk usage (root filesystem)
        disk = psutil.disk_usage("/")
        health_data["disk_percent"] = disk.percent
        health_data["disk_used_gb"] = disk.used / (1024**3)
        health_data["disk_total_gb"] = disk.total / (1024**3)

        # Network connections (basic check)
        net_connections = len(psutil.net_connections())
        health_data["network_connections"] = net_connections

        # Process count
        health_data["process_count"] = len(psutil.pids())

        # Load average (Unix-like systems)
        try:
            load_avg = psutil.getloadavg()
            health_data["load_average"] = load_avg
        except AttributeError:
            # Windows doesn't have getloadavg
            health_data["load_average"] = None

    except Exception as e:
        logger.error(f"Error collecting system metrics: {e}")
        # Provide fallback values
        health_data.update(
            {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_percent": 0.0,
                "error": str(e),
            }
        )

    return health_data


def get_system_health() -> Dict[str, Any]:
    """
    Get current system health status.

    This can be called by health check endpoints or monitoring systems.

    Returns:
        Dict with current health status
    """
    try:
        health_data = _perform_health_checks()

        # Determine overall health status
        is_healthy = (
            health_data.get("cpu_percent", 100) < 95
            and health_data.get("memory_percent", 100) < 95
            and health_data.get("disk_percent", 100) < 95
        )

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": time.time(),
            "metrics": health_data,
            "checks": {
                "cpu": health_data.get("cpu_percent", 0) < 95,
                "memory": health_data.get("memory_percent", 0) < 95,
                "disk": health_data.get("disk_percent", 0) < 95,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        return {"status": "error", "error": str(e), "timestamp": time.time()}
