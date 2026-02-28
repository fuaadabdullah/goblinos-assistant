"""
Latency Monitoring Service
Tracks response times, token usage, and health metrics for latency-aware routing.
Uses Redis for caching and fast lookups.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import redis.asyncio as redis
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetric:
    """Represents a latency measurement."""

    provider_name: str
    model_name: str
    response_time_ms: float
    tokens_used: int
    timestamp: datetime
    success: bool
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatencyMetric":
        """Create from dictionary (Redis retrieval)."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ProviderHealth:
    """Represents current health status of a provider."""

    provider_name: str
    model_name: str
    avg_response_time_ms: float
    success_rate: float
    tokens_per_second: float
    last_updated: datetime
    sample_count: int
    is_healthy: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        data["last_updated"] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderHealth":
        """Create from dictionary (Redis retrieval)."""
        data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


class LatencyMonitoringService:
    """Service for monitoring latency, throughput, and health of LLM providers."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize latency monitoring service.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None

        # Configuration
        self.metrics_ttl = 3600 * 24  # 24 hours
        self.health_ttl = 3600 * 6  # 6 hours
        self.max_samples_per_provider = 1000
        self.health_update_interval = 60  # seconds

        # Latency thresholds (configurable)
        self.latency_thresholds = {
            "ultra_low": 500,  # ms
            "low": 1000,  # ms
            "medium": 2000,  # ms
            "high": 5000,  # ms
        }

        # Health thresholds
        self.health_thresholds = {
            "min_success_rate": 0.95,  # 95% success rate
            "max_avg_latency": 3000,  # 3 seconds average
            "min_samples": 5,  # Minimum samples for health calculation
        }

    async def connect(self):
        """Connect to Redis."""
        if self.redis is None:
            self.redis = redis.Redis.from_url(self.redis_url)
            try:
                await self.redis.ping()
                logger.info("Connected to Redis for latency monitoring")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis = None

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            self.redis = None

    async def record_metric(
        self,
        provider_name: str,
        model_name: str,
        response_time_ms: float,
        tokens_used: int,
        success: bool = True,
        error_type: Optional[str] = None,
    ) -> None:
        """Record a latency metric.

        Args:
            provider_name: Name of the provider (e.g., 'ollama', 'openai')
            model_name: Name of the model (e.g., 'mistral:7b', 'gpt-4')
            response_time_ms: Response time in milliseconds
            tokens_used: Number of tokens used
            success: Whether the request was successful
            error_type: Type of error if request failed
        """
        if not self.redis:
            await self.connect()
            if not self.redis:
                logger.warning("Redis not available, skipping metric recording")
                return

        metric = LatencyMetric(
            provider_name=provider_name,
            model_name=model_name,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            timestamp=datetime.utcnow(),
            success=success,
            error_type=error_type,
        )

        # Store metric in Redis sorted set (by timestamp)
        key = f"latency:metrics:{provider_name}:{model_name}"
        score = metric.timestamp.timestamp()
        value = json.dumps(metric.to_dict())

        try:
            # Add to sorted set
            await self.redis.zadd(key, {value: score})

            # Trim to keep only recent samples
            await self.redis.zremrangebyrank(
                key, 0, -(self.max_samples_per_provider + 1)
            )

            # Set TTL
            await self.redis.expire(key, self.metrics_ttl)

            # Update health status
            await self._update_provider_health(provider_name, model_name)

            logger.debug(
                f"Recorded latency metric: {provider_name}/{model_name} - "
                f"{response_time_ms:.1f}ms, {tokens_used} tokens, success={success}"
            )

        except Exception as e:
            logger.error(f"Failed to record latency metric: {e}")

    async def get_provider_health(
        self, provider_name: str, model_name: str
    ) -> Optional[ProviderHealth]:
        """Get current health status for a provider/model combination.

        Args:
            provider_name: Name of the provider
            model_name: Name of the model

        Returns:
            ProviderHealth object or None if no data available
        """
        if not self.redis:
            await self.connect()
            if not self.redis:
                return None

        key = f"latency:health:{provider_name}:{model_name}"

        try:
            data = await self.redis.get(key)
            if data:
                health_dict = json.loads(data)
                return ProviderHealth.from_dict(health_dict)
        except Exception as e:
            logger.error(f"Failed to get provider health: {e}")

        return None

    async def get_latency_percentiles(
        self, provider_name: str, model_name: str, hours: int = 1
    ) -> Dict[str, float]:
        """Get latency percentiles for a provider/model over the last N hours.

        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            hours: Number of hours to look back

        Returns:
            Dictionary with p50, p95, p99 latency values
        """
        if not self.redis:
            await self.connect()
            if not self.redis:
                return {}

        key = f"latency:metrics:{provider_name}:{model_name}"
        cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).timestamp()

        try:
            # Get metrics from the last N hours
            metrics_data = await self.redis.zrangebyscore(
                key, cutoff_time, float("inf"), withscores=False
            )

            if not metrics_data:
                return {}

            # Parse response times
            response_times = []
            for data in metrics_data:
                try:
                    metric_dict = json.loads(data)
                    metric = LatencyMetric.from_dict(metric_dict)
                    if metric.success:  # Only count successful requests
                        response_times.append(metric.response_time_ms)
                except Exception as e:
                    logger.warning(f"Failed to parse metric data: {e}")
                    continue

            if not response_times:
                return {}

            # Sort for percentile calculation
            response_times.sort()

            def percentile(p: float) -> float:
                index = int(len(response_times) * p / 100)
                return response_times[min(index, len(response_times) - 1)]

            return {
                "p50": percentile(50),
                "p95": percentile(95),
                "p99": percentile(99),
                "count": len(response_times),
            }

        except Exception as e:
            logger.error(f"Failed to get latency percentiles: {e}")
            return {}

    async def check_sla_compliance(
        self,
        provider_name: str,
        model_name: str,
        target_latency_ms: float,
        min_success_rate: float = 0.95,
    ) -> Dict[str, Any]:
        """Check if a provider meets SLA requirements.

        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            target_latency_ms: Target response time in milliseconds
            min_success_rate: Minimum success rate required

        Returns:
            Dictionary with SLA compliance status and metrics
        """
        health = await self.get_provider_health(provider_name, model_name)
        percentiles = await self.get_latency_percentiles(
            provider_name, model_name, hours=1
        )

        if not health or not percentiles:
            return {
                "compliant": False,
                "reason": "No health data available",
                "data_available": False,
            }

        # Check latency SLA (p95 should be below target)
        latency_compliant = percentiles.get("p95", float("inf")) <= target_latency_ms

        # Check success rate SLA
        success_compliant = health.success_rate >= min_success_rate

        # Overall compliance
        compliant = latency_compliant and success_compliant

        return {
            "compliant": compliant,
            "latency_compliant": latency_compliant,
            "success_compliant": success_compliant,
            "current_p95": percentiles.get("p95"),
            "target_latency": target_latency_ms,
            "current_success_rate": health.success_rate,
            "min_success_rate": min_success_rate,
            "data_available": True,
        }

    async def _update_provider_health(
        self, provider_name: str, model_name: str
    ) -> None:
        """Update health status for a provider/model combination."""
        if not self.redis:
            return

        try:
            # Retrieve recent metrics
            metrics_data = await self._retrieve_recent_metrics(
                provider_name, model_name
            )
            if not metrics_data:
                return

            # Parse metrics data
            parsed_metrics = self._parse_metrics_data(metrics_data)
            if not parsed_metrics:
                return

            # Calculate health metrics
            health_metrics = self._calculate_health_metrics(parsed_metrics)

            # Determine health status
            is_healthy = self._determine_health_status(health_metrics)

            # Store health data
            await self._store_health_data(
                provider_name, model_name, health_metrics, is_healthy
            )

        except Exception as e:
            logger.error(f"Failed to update provider health: {e}")

    async def _retrieve_recent_metrics(
        self, provider_name: str, model_name: str
    ) -> Optional[List[str]]:
        """Retrieve recent metrics data from Redis."""
        key = f"latency:metrics:{provider_name}:{model_name}"
        cutoff_time = (datetime.utcnow() - timedelta(hours=1)).timestamp()

        metrics_data = await self.redis.zrangebyscore(
            key, cutoff_time, float("inf"), withscores=False
        )

        if len(metrics_data) < self.health_thresholds["min_samples"]:
            return None  # Not enough data for reliable health calculation

        return metrics_data

    def _parse_metrics_data(self, metrics_data: List[str]) -> Optional[Dict[str, Any]]:
        """Parse metrics data and extract relevant statistics."""
        response_times = []
        total_tokens = 0
        success_count = 0
        total_count = len(metrics_data)

        for data in metrics_data:
            try:
                metric_dict = json.loads(data)
                metric = LatencyMetric.from_dict(metric_dict)
                response_times.append(metric.response_time_ms)
                total_tokens += metric.tokens_used
                if metric.success:
                    success_count += 1
            except Exception as e:
                logger.warning(f"Failed to parse metric for health calculation: {e}")
                continue

        if not response_times:
            return None

        return {
            "response_times": response_times,
            "total_tokens": total_tokens,
            "success_count": success_count,
            "total_count": total_count,
        }

    def _calculate_health_metrics(
        self, parsed_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate health metrics from parsed data."""
        response_times = parsed_metrics["response_times"]
        total_tokens = parsed_metrics["total_tokens"]
        success_count = parsed_metrics["success_count"]
        total_count = parsed_metrics["total_count"]

        avg_response_time = sum(response_times) / len(response_times)
        success_rate = success_count / total_count if total_count > 0 else 0

        # Calculate tokens per second (throughput)
        total_time_seconds = sum(response_times) / 1000  # Convert to seconds
        tokens_per_second = (
            total_tokens / total_time_seconds if total_time_seconds > 0 else 0
        )

        return {
            "avg_response_time": avg_response_time,
            "success_rate": success_rate,
            "tokens_per_second": tokens_per_second,
            "total_count": total_count,
        }

    def _determine_health_status(self, health_metrics: Dict[str, float]) -> bool:
        """Determine if the provider is healthy based on metrics."""
        return (
            health_metrics["success_rate"] >= self.health_thresholds["min_success_rate"]
            and health_metrics["avg_response_time"]
            <= self.health_thresholds["max_avg_latency"]
            and health_metrics["total_count"] >= self.health_thresholds["min_samples"]
        )

    async def _store_health_data(
        self,
        provider_name: str,
        model_name: str,
        health_metrics: Dict[str, float],
        is_healthy: bool,
    ) -> None:
        """Store health data in Redis."""
        health_key = f"latency:health:{provider_name}:{model_name}"

        health = ProviderHealth(
            provider_name=provider_name,
            model_name=model_name,
            avg_response_time_ms=health_metrics["avg_response_time"],
            success_rate=health_metrics["success_rate"],
            tokens_per_second=health_metrics["tokens_per_second"],
            last_updated=datetime.utcnow(),
            sample_count=int(health_metrics["total_count"]),
            is_healthy=is_healthy,
        )

        await self.redis.set(
            health_key, json.dumps(health.to_dict()), ex=self.health_ttl
        )

    async def get_all_provider_health(self) -> List[ProviderHealth]:
        """Get health status for all providers/models."""
        if not self.redis:
            await self.connect()
            if not self.redis:
                return []

        try:
            # Get all health keys
            pattern = "latency:health:*"
            keys = await self.redis.keys(pattern)

            health_list = []
            for key in keys:
                try:
                    data = await self.redis.get(key)
                    if data:
                        health_dict = json.loads(data)
                        health_list.append(ProviderHealth.from_dict(health_dict))
                except Exception as e:
                    logger.warning(f"Failed to parse health data for key {key}: {e}")
                    continue

            return health_list

        except Exception as e:
            logger.error(f"Failed to get all provider health: {e}")
            return []

    async def cleanup_old_metrics(self, days: int = 7) -> int:
        """Clean up metrics older than specified days.

        Args:
            days: Number of days to keep metrics

        Returns:
            Number of metrics cleaned up
        """
        if not self.redis:
            await self.connect()
            if not self.redis:
                return 0

        cutoff_time = (datetime.utcnow() - timedelta(days=days)).timestamp()
        cleaned_count = 0

        try:
            # Get all metrics keys
            pattern = "latency:metrics:*"
            keys = await self.redis.keys(pattern)

            for key in keys:
                # Remove old entries
                removed = await self.redis.zremrangebyscore(key, 0, cutoff_time)
                cleaned_count += removed

            logger.info(f"Cleaned up {cleaned_count} old latency metrics")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
            return 0
