"""
Health and metrics calculations for provider routing.

Handles provider health scoring, response time analysis, and metric aggregation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Any

logger = logging.getLogger(__name__)


async def fetch_recent_provider_metrics(db_session, provider_id: int, hours: int = 1):
    """
    Fetch recent metrics for a provider from the database.

    Args:
        db_session: Database session
        provider_id: Provider ID to fetch metrics for
        hours: Number of hours to look back

    Returns:
        List of ProviderMetric objects
    """
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)

    def _sync_get_metrics():
        from models import ProviderMetric

        return (
            db_session.query(ProviderMetric)
            .filter(
                ProviderMetric.provider_id == provider_id,
                ProviderMetric.timestamp >= cutoff_time,
            )
            .all()
        )

    return await asyncio.to_thread(_sync_get_metrics)


async def fetch_recent_provider_metrics_limited(
    db_session, provider_id: int, limit: int = 10, hours: int = 1
):
    """
    Fetch recent metrics for a provider with a limit.

    Args:
        db_session: Database session
        provider_id: Provider ID to fetch metrics for
        limit: Maximum number of metrics to fetch
        hours: Number of hours to look back

    Returns:
        List of ProviderMetric objects
    """
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)

    def _sync_get_recent_metrics():
        from models import ProviderMetric

        return (
            db_session.query(ProviderMetric)
            .filter(
                ProviderMetric.provider_id == provider_id,
                ProviderMetric.timestamp >= cutoff_time,
            )
            .order_by(ProviderMetric.timestamp.desc())
            .limit(limit)
            .all()
        )

    return await asyncio.to_thread(_sync_get_recent_metrics)


def calculate_health_rate(metrics: List[Any]) -> float:
    """
    Calculate the health rate from provider metrics.

    Args:
        metrics: List of ProviderMetric objects

    Returns:
        Health rate (0.0 to 1.0)
    """
    if not metrics:
        return 0.0

    healthy_count = sum(1 for m in metrics if m.is_healthy)
    return healthy_count / len(metrics)


def calculate_average_response_time(metrics: List[Any]) -> float:
    """
    Calculate average response time from provider metrics.

    Args:
        metrics: List of ProviderMetric objects

    Returns:
        Average response time in milliseconds
    """
    response_times = [m.response_time_ms for m in metrics if m.response_time_ms]
    if not response_times:
        return 1000.0  # Default fallback

    return sum(response_times) / len(response_times)


def calculate_response_time_score(avg_response_time: float) -> float:
    """
    Calculate response time score based on average response time.

    Args:
        avg_response_time: Average response time in milliseconds

    Returns:
        Response time score (0-25, higher is better)
    """
    # Response time score (faster = better, max 2000ms = 0 points)
    return max(0, 25 - (avg_response_time / 80))


def calculate_overall_health_score(
    health_rate: float, response_time_score: float
) -> float:
    """
    Calculate overall health score combining health rate and response time.

    Args:
        health_rate: Health rate (0.0 to 1.0)
        response_time_score: Response time score

    Returns:
        Overall health score (-50 to 75)
    """
    # Health score: -50 (all unhealthy) to 50 (all healthy)
    health_score = (health_rate - 0.5) * 100
    return health_score + response_time_score


async def calculate_provider_health_score(db_session, provider_id: int) -> float:
    """
    Calculate comprehensive health score for a provider.

    Args:
        db_session: Database session
        provider_id: Provider ID

    Returns:
        Health score (-50 to 75)
    """
    metrics = await fetch_recent_provider_metrics(db_session, provider_id)

    if not metrics:
        return 0.0  # No data = neutral

    health_rate = calculate_health_rate(metrics)
    avg_response_time = calculate_average_response_time(metrics)
    response_time_score = calculate_response_time_score(avg_response_time)

    return calculate_overall_health_score(health_rate, response_time_score)


def calculate_performance_bonus_from_response_time(avg_response_time: float) -> float:
    """
    Calculate performance bonus based on average response time.

    Args:
        avg_response_time: Average response time in milliseconds

    Returns:
        Performance bonus (0-15)
    """
    # Bonus for faster response times (under 500ms = 15 points, over 2000ms = 0)
    if avg_response_time <= 500:
        return 15.0
    elif avg_response_time >= 2000:
        return 0.0
    else:
        return 15.0 * (2000 - avg_response_time) / 1500


async def calculate_provider_performance_bonus(db_session, provider_id: int) -> float:
    """
    Calculate performance bonus for a provider based on recent metrics.

    Args:
        db_session: Database session
        provider_id: Provider ID

    Returns:
        Performance bonus (0-15)
    """
    metrics = await fetch_recent_provider_metrics_limited(db_session, provider_id)

    if not metrics:
        return 0.0

    response_times = [m.response_time_ms for m in metrics if m.response_time_ms]
    if not response_times:
        return 0.0

    avg_response_time = sum(response_times) / len(response_times)

    return calculate_performance_bonus_from_response_time(avg_response_time)
