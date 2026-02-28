"""
Simple Redis-backed Circuit Breaker for external API calls.

Prevents ping-pong retries and cascade failures by tracking failures
and temporarily blocking calls to failing providers.
"""

import redis
import time
import logging
import os
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """Simple Redis-backed circuit breaker."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 3,
        redis_client: Optional[redis.Redis] = None,
        timeout: float = 30.0,
    ):
        """Initialize circuit breaker.

        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            success_threshold: Number of successes needed to close circuit
            redis_client: Redis client for distributed state
            timeout: Timeout for Redis operations
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout = timeout

        # Redis keys
        self.fail_key = f"circuit:{name}:failures"
        self.success_key = f"circuit:{name}:successes"
        self.state_key = f"circuit:{name}:state"
        self.last_fail_key = f"circuit:{name}:last_fail"

        # Initialize Redis client
        if redis_client is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            try:
                self.redis = redis.from_url(redis_url, decode_responses=True)
                self.redis.ping()  # Test connection
            except Exception as e:
                logger.warning(
                    f"Redis connection failed, using in-memory fallback: {e}"
                )
                self.redis = None
        else:
            self.redis = redis_client

        # In-memory fallback state (used when Redis is unavailable).
        # This is per-process and not shared across instances.
        self._mem_state: str = "closed"
        self._mem_failures: int = 0
        self._mem_successes: int = 0
        self._mem_last_fail: Optional[float] = None

    def _get_state(self) -> str:
        """Get current circuit state."""
        if self.redis is None:
            return self._mem_state

        try:
            state = self.redis.get(self.state_key)
            return state or "closed"
        except Exception as e:
            logger.warning(f"Redis error getting state: {e}")
            return "closed"

    def _set_state(self, state: str):
        """Set circuit state."""
        if self.redis is None:
            self._mem_state = state
            return

        try:
            self.redis.set(self.state_key, state)
        except Exception as e:
            logger.warning(f"Redis error setting state: {e}")

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.redis is None:
            if self._mem_last_fail is None:
                return False
            return time.time() - float(self._mem_last_fail) > self.recovery_timeout

        try:
            last_fail = self.redis.get(self.last_fail_key)
            if last_fail is None:
                return False

            return time.time() - float(last_fail) > self.recovery_timeout
        except Exception as e:
            logger.warning(f"Redis error checking reset: {e}")
            return False

    def _record_attempt(self, success: bool):
        """Record a call attempt."""
        if self.redis is None:
            if success:
                # Reset failure/success counts on success
                self._mem_failures = 0
                self._mem_successes = 0
                self._mem_state = "closed"
                return

            # Failure path
            self._mem_failures += 1
            self._mem_last_fail = time.time()
            if self._mem_failures >= self.failure_threshold:
                self._mem_state = "open"
            return

        try:
            if success:
                # Reset failure count on success
                self.redis.delete(self.fail_key)
                self.redis.delete(self.success_key)
                self._set_state("closed")
            else:
                # Increment failure count
                failures = self.redis.incr(self.fail_key)
                self.redis.set(self.last_fail_key, time.time())

                if failures >= self.failure_threshold:
                    self._set_state("open")
        except Exception as e:
            logger.warning(f"Redis error recording attempt: {e}")

    def before_call(self):
        """Check if call should proceed."""
        state = self._get_state()

        if state == "open":
            if self._should_attempt_reset():
                # Try half-open state
                self._set_state("half-open")
                return True
            else:
                raise CircuitBreakerOpen(f"Circuit {self.name} is open")

        return True

    def record_success(self):
        """Record successful call."""
        state = self._get_state()

        if state == "half-open":
            # Increment success count
            if self.redis:
                try:
                    successes = self.redis.incr(self.success_key)
                    if successes >= self.success_threshold:
                        self._set_state("closed")
                        self.redis.delete(self.fail_key)
                        self.redis.delete(self.success_key)
                except Exception as e:
                    logger.warning(f"Redis error recording success: {e}")
            else:
                self._mem_successes += 1
                if self._mem_successes >= self.success_threshold:
                    self._mem_state = "closed"
                    self._mem_failures = 0
                    self._mem_successes = 0
        else:
            self._record_attempt(True)

    def record_failure(self):
        """Record failed call."""
        self._record_attempt(False)

    def __call__(self, func: Callable, *args, **kwargs) -> Any:
        """Decorator to wrap function calls with circuit breaker."""
        self.before_call()

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise

    async def acall(self, func: Callable, *args, **kwargs) -> Any:
        """Async decorator to wrap function calls with circuit breaker."""
        self.before_call()

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise

    def get_status(self) -> dict:
        """Get circuit breaker status."""
        state = self._get_state()

        status = {
            "state": state,
            "name": self.name,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "success_threshold": self.success_threshold,
        }

        if self.redis:
            try:
                failures = self.redis.get(self.fail_key)
                successes = self.redis.get(self.success_key)
                last_fail = self.redis.get(self.last_fail_key)

                status.update(
                    {
                        "current_failures": int(failures) if failures else 0,
                        "current_successes": int(successes) if successes else 0,
                        "last_failure_time": float(last_fail) if last_fail else None,
                    }
                )
            except Exception as e:
                logger.warning(f"Redis error getting status: {e}")
        else:
            status.update(
                {
                    "current_failures": int(self._mem_failures),
                    "current_successes": int(self._mem_successes),
                    "last_failure_time": float(self._mem_last_fail)
                    if self._mem_last_fail
                    else None,
                }
            )

        return status


# Global circuit breaker registry
_circuit_breakers = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    success_threshold: int = 3,
    timeout: float = 30.0,
) -> CircuitBreaker:
    """Get or create a circuit breaker instance."""
    key = f"{name}:{failure_threshold}:{recovery_timeout}:{success_threshold}"

    if key not in _circuit_breakers:
        _circuit_breakers[key] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            timeout=timeout,
        )

    return _circuit_breakers[key]
