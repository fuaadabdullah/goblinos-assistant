"""
ChatRetryHandler Service for handling retry logic.

This service handles retry logic for failed chat completion requests,
separating retry concerns from the main chat handler.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Types of retry strategies."""

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    JITTERED = "jittered"


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter_factor: float = 0.1  # For jittered strategy
    retryable_errors: Optional[List[str]] = None
    backoff_multiplier: float = 2.0
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 300  # 5 minutes


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""

    attempt_number: int
    delay_seconds: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: float = 0.0


class CircuitBreakerState(Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service is recovered


class CircuitBreaker:
    """Circuit breaker implementation for retry logic."""

    def __init__(self, config: RetryConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs):
        """Execute a function with circuit breaker protection."""
        async with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if (
                    time.time() - self.last_failure_time
                    > self.config.circuit_breaker_recovery_timeout
                ):
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise RuntimeError("Circuit breaker is OPEN")

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e

    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to close
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if (
            self.state == CircuitBreakerState.CLOSED
            and self.failure_count >= self.config.circuit_breaker_failure_threshold
        ):
            self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.failure_count = 1


class ChatRetryHandler:
    """Service for handling retry logic."""

    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize the ChatRetryHandler."""
        self.config = config or RetryConfig()
        self.circuit_breaker = (
            CircuitBreaker(self.config) if self.config.enable_circuit_breaker else None
        )
        self.retry_history: Dict[str, List[RetryAttempt]] = {}

        # Statistics
        self.stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "circuit_breaker_trips": 0,
            "circuit_breaker_recoveries": 0,
        }

    async def retry_with_strategy(
        self,
        func: Callable,
        *args,
        request_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Arguments for the function
            request_id: Request ID for tracking
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function execution

        Raises:
            Exception: If all retry attempts fail
        """
        if request_id:
            self.retry_history[request_id] = []

        last_exception: Optional[BaseException] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self._execute_with_circuit_breaker(func, *args, **kwargs)
                self._handle_successful_attempt(request_id, attempt)
                return result

            except Exception as e:
                last_exception = e
                should_retry = self._should_retry_attempt(e, attempt)
                if not should_retry:
                    break

                delay = self._calculate_delay(attempt + 1)
                self._record_failed_attempt(request_id, attempt, delay, e)
                await asyncio.sleep(delay)

        # All retries failed
        logger.error(f"All retry attempts failed for request {request_id}")
        if last_exception is not None:
            raise last_exception
        raise RuntimeError(f"All retry attempts failed for request {request_id}")

    async def _execute_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker if enabled."""
        if self.circuit_breaker:
            return await self.circuit_breaker.call(func, *args, **kwargs)
        else:
            return await func(*args, **kwargs)

    def _handle_successful_attempt(self, request_id: Optional[str], attempt: int):
        """Handle successful retry attempt."""
        if request_id:
            self._record_retry_attempt(request_id, attempt + 1, 0.0, None, None)

        if attempt > 0:
            self.stats["successful_retries"] += 1

        self.stats["total_retries"] += 1

    def _should_retry_attempt(self, error: Exception, attempt: int) -> bool:
        """Determine if we should retry after an error."""
        self.stats["total_retries"] += 1
        self.stats["failed_retries"] += 1

        if not self._is_retryable_error(error):
            return False

        if attempt >= self.config.max_retries:
            return False

        return True

    def _record_failed_attempt(
        self, request_id: Optional[str], attempt: int, delay: float, error: Exception
    ):
        """Record a failed retry attempt."""
        if request_id:
            self._record_retry_attempt(
                request_id,
                attempt + 1,
                delay,
                type(error).__name__,
                str(error),
            )

        logger.warning(
            f"Retry attempt {attempt + 1}/{self.config.max_retries + 1} failed for request {request_id}: {error}"
        )
        logger.debug(f"Waiting {delay:.2f} seconds before retry")

    def retry_decorator(self, request_id_param: str = "request_id"):
        """
        Decorator for retry logic.

        Args:
            request_id_param: Parameter name containing request_id

        Returns:
            Decorator function
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request_id from kwargs or args
                request_id = kwargs.get(request_id_param)
                if request_id is None and args:
                    # Try to get from first argument if it's a method
                    if hasattr(args[0], "request_id"):
                        request_id = getattr(args[0], "request_id", None)

                return await self.retry_with_strategy(
                    func, *args, request_id=request_id, **kwargs
                )

            return wrapper

        return decorator

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        error_type = type(error).__name__

        # If no retryable errors specified, assume all are retryable
        if not self.config.retryable_errors:
            return True

        return error_type in self.config.retryable_errors

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED:
            return self.config.base_delay_seconds

        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay_seconds * attempt

        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay_seconds * (
                self.config.backoff_multiplier ** (attempt - 1)
            )

        elif self.config.strategy == RetryStrategy.JITTERED:
            base_delay = self.config.base_delay_seconds * (
                self.config.backoff_multiplier ** (attempt - 1)
            )
            jitter = base_delay * self.config.jitter_factor
            import random

            delay = base_delay + random.uniform(-jitter, jitter)
            delay = max(0, delay)

        else:
            delay = self.config.base_delay_seconds

        # Cap the delay at max_delay_seconds
        return min(delay, self.config.max_delay_seconds)

    def _record_retry_attempt(
        self,
        request_id: str,
        attempt_number: int,
        delay_seconds: float,
        error_type: Optional[str],
        error_message: Optional[str],
    ) -> None:
        """Record a retry attempt."""
        attempt = RetryAttempt(
            attempt_number=attempt_number,
            delay_seconds=delay_seconds,
            error_type=error_type,
            error_message=error_message,
            timestamp=time.time(),
        )

        if request_id not in self.retry_history:
            self.retry_history[request_id] = []

        self.retry_history[request_id].append(attempt)

    def get_retry_history(self, request_id: str) -> List[RetryAttempt]:
        """Get retry history for a request."""
        return self.retry_history.get(request_id, [])

    def get_retry_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        try:
            current_time = time.time()

            # Calculate success rate
            total_attempts = self.stats["total_retries"]
            success_rate = self.stats["successful_retries"] / max(total_attempts, 1)
            failure_rate = self.stats["failed_retries"] / max(total_attempts, 1)

            # Get circuit breaker status
            cb_status = None
            if self.circuit_breaker:
                cb_status = {
                    "state": self.circuit_breaker.state.value,
                    "failure_count": self.circuit_breaker.failure_count,
                    "success_count": self.circuit_breaker.success_count,
                    "last_failure_time": self.circuit_breaker.last_failure_time,
                }

            return {
                "total_retries": total_attempts,
                "successful_retries": self.stats["successful_retries"],
                "failed_retries": self.stats["failed_retries"],
                "success_rate": success_rate,
                "failure_rate": failure_rate,
                "circuit_breaker_trips": self.stats["circuit_breaker_trips"],
                "circuit_breaker_recoveries": self.stats["circuit_breaker_recoveries"],
                "circuit_breaker_status": cb_status,
                "config": {
                    "max_retries": self.config.max_retries,
                    "base_delay_seconds": self.config.base_delay_seconds,
                    "max_delay_seconds": self.config.max_delay_seconds,
                    "strategy": self.config.strategy.value,
                    "jitter_factor": self.config.jitter_factor,
                    "retryable_errors": self.config.retryable_errors,
                    "backoff_multiplier": self.config.backoff_multiplier,
                    "enable_circuit_breaker": self.config.enable_circuit_breaker,
                    "circuit_breaker_failure_threshold": self.config.circuit_breaker_failure_threshold,
                    "circuit_breaker_recovery_timeout": self.config.circuit_breaker_recovery_timeout,
                },
                "timestamp": current_time,
            }

        except Exception as e:
            logger.error(f"Error getting retry stats: {e}")
            return {"error": str(e)}

    def get_retry_health(self) -> Dict[str, Any]:
        """Get retry health information."""
        try:
            stats = self.get_retry_stats()
            current_time = time.time()

            issues = self._check_health_issues(stats)

            return {
                "status": "healthy" if not issues else "warning",
                "issues": issues,
                "stats": stats,
                "timestamp": current_time,
            }

        except Exception as e:
            logger.error(f"Error getting retry health: {e}")
            return {"status": "error", "error": str(e)}

    def _check_health_issues(self, stats: Dict[str, Any]) -> List[str]:
        """Check for potential health issues in retry statistics."""
        issues: List[str] = []

        self._check_success_rate(stats, issues)
        self._check_circuit_breaker_status(stats, issues)
        self._check_failure_rate(stats, issues)

        return issues

    def _check_success_rate(self, stats: Dict[str, Any], issues: List[str]):
        """Check if success rate is too low."""
        if stats["success_rate"] < 0.5:  # Less than 50% success rate
            issues.append(f"Low retry success rate: {stats['success_rate']:.2%}")

    def _check_circuit_breaker_status(self, stats: Dict[str, Any], issues: List[str]):
        """Check if circuit breaker is open."""
        if stats.get("circuit_breaker_status", {}).get("state") == "open":
            issues.append("Circuit breaker is OPEN")

    def _check_failure_rate(self, stats: Dict[str, Any], issues: List[str]):
        """Check if failure rate is too high."""
        if stats["failure_rate"] > 0.2:  # More than 20% failure rate
            issues.append(f"High failure rate: {stats['failure_rate']:.2%}")

    def update_retry_config(self, config: RetryConfig) -> None:
        """Update retry configuration."""
        self.config = config
        if config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(config)
        else:
            self.circuit_breaker = None
        logger.info("Updated retry configuration")

    def reset_stats(self) -> None:
        """Reset retry statistics."""
        try:
            self.stats = {
                "total_retries": 0,
                "successful_retries": 0,
                "failed_retries": 0,
                "circuit_breaker_trips": 0,
                "circuit_breaker_recoveries": 0,
            }
            self.retry_history.clear()
            logger.info("Reset retry statistics")

        except Exception as e:
            logger.error(f"Error resetting retry stats: {e}")

    def get_retry_recommendations(self) -> List[str]:
        """Get recommendations for retry configuration."""
        try:
            stats = self.get_retry_stats()
            recommendations = []

            # Low success rate
            if stats["success_rate"] < 0.3:
                recommendations.append(
                    "Consider increasing max_retries or base_delay_seconds"
                )

            # High failure rate
            if stats["failure_rate"] > 0.5:
                recommendations.append(
                    "Consider reviewing retryable_errors configuration"
                )

            # Circuit breaker frequently tripping
            if stats["circuit_breaker_trips"] > 10:
                recommendations.append("Consider adjusting circuit breaker thresholds")

            # No recommendations
            if not recommendations:
                recommendations.append("Retry configuration appears optimal")

            return recommendations

        except Exception as e:
            logger.error(f"Error getting retry recommendations: {e}")
            return ["Error calculating recommendations"]

    def clear_retry_history(self, request_id: Optional[str] = None) -> int:
        """
        Clear retry history.

        Args:
            request_id: Specific request ID to clear (optional)

        Returns:
            Number of histories cleared
        """
        try:
            if request_id:
                if request_id in self.retry_history:
                    del self.retry_history[request_id]
                    return 1
                return 0
            else:
                count = len(self.retry_history)
                self.retry_history.clear()
                return count

        except Exception as e:
            logger.error(f"Error clearing retry history: {e}")
            return 0

    def get_retry_analysis(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get detailed retry analysis for a time window.

        Args:
            time_window_hours: Time window in hours

        Returns:
            Retry analysis data
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (time_window_hours * 3600)

            analysis = self._analyze_retry_history(cutoff_time)

            return {
                "time_window_hours": time_window_hours,
                "total_attempts": analysis["total_attempts"],
                "successful_attempts": analysis["successful_attempts"],
                "failed_attempts": analysis["failed_attempts"],
                "success_rate": analysis["successful_attempts"]
                / max(analysis["total_attempts"], 1),
                "average_delay_seconds": analysis["average_delay"],
                "total_delay_seconds": analysis["total_delay"],
                "most_common_errors": analysis["most_common_errors"],
                "error_distribution": analysis["error_types"],
                "timestamp": current_time,
            }

        except Exception as e:
            logger.error(f"Error getting retry analysis: {e}")
            return {"error": str(e)}

    def _analyze_retry_history(self, cutoff_time: float) -> Dict[str, Any]:
        """Analyze retry history within the time window."""
        analysis_data = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
            "error_types": {},
            "total_delay": 0.0,
        }

        for request_attempts in self.retry_history.values():
            self._process_request_attempts(request_attempts, cutoff_time, analysis_data)

        total_attempts = int(analysis_data.get("total_attempts", 0))
        total_delay = float(analysis_data.get("total_delay", 0.0))
        error_types_raw = analysis_data.get("error_types", {})
        error_types = dict(error_types_raw) if isinstance(error_types_raw, dict) else {}

        average_delay = total_delay / total_attempts if total_attempts > 0 else 0.0

        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_attempts": analysis_data["total_attempts"],
            "successful_attempts": analysis_data["successful_attempts"],
            "failed_attempts": analysis_data["failed_attempts"],
            "average_delay": average_delay,
            "total_delay": analysis_data["total_delay"],
            "most_common_errors": sorted_errors[:5],
            "error_types": analysis_data["error_types"],
        }

    def _process_request_attempts(
        self,
        request_attempts: List[RetryAttempt],
        cutoff_time: float,
        analysis_data: Dict[str, Any],
    ):
        """Process attempts for a single request."""
        for attempt in request_attempts:
            if attempt.timestamp < cutoff_time:
                continue

            analysis_data["total_attempts"] += 1
            analysis_data["total_delay"] += attempt.delay_seconds

            if attempt.error_type is None:
                analysis_data["successful_attempts"] += 1
            else:
                analysis_data["failed_attempts"] += 1
                error_type = attempt.error_type
                analysis_data["error_types"][error_type] = (
                    analysis_data["error_types"].get(error_type, 0) + 1
                )

    async def test_retry_strategy(
        self, func: Callable, *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Test retry strategy with a function.

        Args:
            func: Function to test
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Test results
        """
        try:
            start_time = time.time()

            try:
                await self.retry_with_strategy(func, *args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)

            end_time = time.time()

            return {
                "success": success,
                "error": error,
                "execution_time": end_time - start_time,
                "stats": self.get_retry_stats(),
            }

        except Exception as e:
            logger.error(f"Error testing retry strategy: {e}")
            return {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retry handler statistics.

        Returns:
            Dict containing retry statistics
        """
        total_attempts = self.stats["total_retries"] + self.stats["successful_retries"]
        success_rate = (
            self.stats["successful_retries"] / total_attempts
            if total_attempts > 0
            else 0.0
        )
        return {
            "total_retries": self.stats["total_retries"],
            "successful_retries": self.stats["successful_retries"],
            "failed_retries": self.stats["failed_retries"],
            "success_rate": success_rate,
            "circuit_breaker_trips": self.stats["circuit_breaker_trips"],
            "timestamp": time.time(),
        }
