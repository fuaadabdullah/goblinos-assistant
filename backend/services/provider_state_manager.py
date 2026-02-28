"""
Thread-safe provider state manager.

Replaces module-level dictionaries with a centralized, thread-safe state manager
for provider health, circuit breaker, and rate limiting state.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from threading import RLock


@dataclass
class ProviderState:
    """State for a single provider"""

    health_check_time: float = 0.0
    is_healthy: bool = True
    failure_count: int = 0
    circuit_open_until: float = 0.0
    auth_blocked_until: float = 0.0
    rate_limited_until: float = 0.0


class ProviderStateManager:
    """
    Thread-safe manager for provider state tracking.

    Manages health caching, circuit breaker state, failure counting,
    and various blocking conditions for LLM providers.

    Configuration via environment variables:
    - GENERATE_PROVIDER_HEALTH_TTL_S: Health cache TTL (default: 60s)
    - GENERATE_PROVIDER_CIRCUIT_FAILS: Failures before circuit opens (default: 3)
    - GENERATE_PROVIDER_CIRCUIT_COOLDOWN_S: Circuit breaker cooldown (default: 30s)
    - GENERATE_PROVIDER_AUTH_COOLDOWN_S: Auth failure cooldown (default: 600s)
    - GENERATE_PROVIDER_RATE_LIMIT_COOLDOWN_S: Rate limit cooldown (default: 120s)
    """

    def __init__(
        self,
        health_ttl_s: float = 60.0,
        circuit_threshold: int = 3,
        circuit_cooldown_s: float = 30.0,
        auth_cooldown_s: float = 600.0,
        rate_limit_cooldown_s: float = 120.0,
    ):
        """Initialize provider state manager with configuration"""
        self._lock = RLock()
        self._states: Dict[str, ProviderState] = {}

        # Configuration
        self.health_ttl_s = health_ttl_s
        self.circuit_threshold = circuit_threshold
        self.circuit_cooldown_s = circuit_cooldown_s
        self.auth_cooldown_s = auth_cooldown_s
        self.rate_limit_cooldown_s = rate_limit_cooldown_s

    def _get_state(self, provider_id: str) -> ProviderState:
        """Get or create state for provider (caller must hold lock)"""
        if provider_id not in self._states:
            self._states[provider_id] = ProviderState()
        return self._states[provider_id]

    def reset(self) -> None:
        """Reset all provider state (useful for testing)"""
        with self._lock:
            self._states.clear()

    def reset_provider(self, provider_id: str) -> None:
        """Reset state for a specific provider"""
        with self._lock:
            if provider_id in self._states:
                del self._states[provider_id]

    # Health cache operations

    def get_cached_health(self, provider_id: str) -> Optional[Tuple[float, bool]]:
        """
        Get cached health status for provider.

        Returns:
            Tuple of (timestamp, is_healthy) if cached and not expired, None otherwise
        """
        with self._lock:
            state = self._get_state(provider_id)
            current_time = time.time()

            # Check if cache is still valid
            if current_time - state.health_check_time < self.health_ttl_s:
                return (state.health_check_time, state.is_healthy)

            return None

    def set_health_status(self, provider_id: str, is_healthy: bool) -> None:
        """Update health status for provider"""
        with self._lock:
            state = self._get_state(provider_id)
            state.health_check_time = time.time()
            state.is_healthy = is_healthy

    # Failure tracking and circuit breaker

    def record_failure(self, provider_id: str) -> int:
        """
        Record a failure for provider and return current failure count.

        If failure count reaches threshold, opens circuit breaker.

        Returns:
            Current failure count after increment
        """
        with self._lock:
            state = self._get_state(provider_id)
            state.failure_count += 1

            # Open circuit if threshold reached
            if state.failure_count >= self.circuit_threshold:
                state.circuit_open_until = time.time() + self.circuit_cooldown_s

            return state.failure_count

    def record_success(self, provider_id: str) -> None:
        """Record a successful call, resetting failure count"""
        with self._lock:
            state = self._get_state(provider_id)
            state.failure_count = 0
            # Don't close circuit immediately - let cooldown expire naturally

    def get_failure_count(self, provider_id: str) -> int:
        """Get current failure count for provider"""
        with self._lock:
            state = self._get_state(provider_id)
            return state.failure_count

    def is_circuit_open(self, provider_id: str) -> bool:
        """Check if circuit breaker is open for provider"""
        with self._lock:
            state = self._get_state(provider_id)
            current_time = time.time()

            # Circuit is open if cooldown period hasn't expired
            return current_time < state.circuit_open_until

    def close_circuit(self, provider_id: str) -> None:
        """Manually close circuit breaker (useful for testing/recovery)"""
        with self._lock:
            state = self._get_state(provider_id)
            state.circuit_open_until = 0.0
            state.failure_count = 0

    # Authentication blocking

    def block_auth(self, provider_id: str) -> None:
        """Block provider due to authentication failure (401)"""
        with self._lock:
            state = self._get_state(provider_id)
            state.auth_blocked_until = time.time() + self.auth_cooldown_s

    def is_auth_blocked(self, provider_id: str) -> bool:
        """Check if provider is blocked due to auth failure"""
        with self._lock:
            state = self._get_state(provider_id)
            current_time = time.time()
            return current_time < state.auth_blocked_until

    def unblock_auth(self, provider_id: str) -> None:
        """Manually unblock auth (useful for recovery after credentials update)"""
        with self._lock:
            state = self._get_state(provider_id)
            state.auth_blocked_until = 0.0

    # Rate limiting

    def block_rate_limit(self, provider_id: str) -> None:
        """Block provider due to rate limit (429)"""
        with self._lock:
            state = self._get_state(provider_id)
            state.rate_limited_until = time.time() + self.rate_limit_cooldown_s

    def is_rate_limited(self, provider_id: str) -> bool:
        """Check if provider is rate limited"""
        with self._lock:
            state = self._get_state(provider_id)
            current_time = time.time()
            return current_time < state.rate_limited_until

    def unblock_rate_limit(self, provider_id: str) -> None:
        """Manually unblock rate limit (useful for testing)"""
        with self._lock:
            state = self._get_state(provider_id)
            state.rate_limited_until = 0.0

    # Composite checks

    def is_available(self, provider_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if provider is available for use.

        Returns:
            Tuple of (is_available, reason) where reason is None if available
        """
        with self._lock:
            if self.is_auth_blocked(provider_id):
                return False, "auth_blocked"
            if self.is_rate_limited(provider_id):
                return False, "rate_limited"
            if self.is_circuit_open(provider_id):
                return False, "circuit_open"
            return True, None

    def get_provider_status(self, provider_id: str) -> Dict[str, any]:
        """Get complete status for provider (useful for debugging/monitoring)"""
        with self._lock:
            state = self._get_state(provider_id)
            current_time = time.time()

            return {
                "provider_id": provider_id,
                "is_healthy": state.is_healthy,
                "health_cached": current_time - state.health_check_time < self.health_ttl_s,
                "health_age_s": current_time - state.health_check_time,
                "failure_count": state.failure_count,
                "circuit_open": current_time < state.circuit_open_until,
                "circuit_closes_in_s": max(0, state.circuit_open_until - current_time),
                "auth_blocked": current_time < state.auth_blocked_until,
                "auth_unblocks_in_s": max(0, state.auth_blocked_until - current_time),
                "rate_limited": current_time < state.rate_limited_until,
                "rate_limit_ends_in_s": max(0, state.rate_limited_until - current_time),
            }

    def get_all_provider_statuses(self) -> Dict[str, Dict[str, any]]:
        """Get status for all tracked providers"""
        with self._lock:
            return {
                provider_id: self.get_provider_status(provider_id)
                for provider_id in self._states.keys()
            }


# Global instance
_provider_state_manager: Optional[ProviderStateManager] = None


def get_provider_state_manager() -> ProviderStateManager:
    """Get or create the global provider state manager"""
    global _provider_state_manager
    if _provider_state_manager is None:
        import os

        _provider_state_manager = ProviderStateManager(
            health_ttl_s=float(os.getenv("GENERATE_PROVIDER_HEALTH_TTL_S", "60")),
            circuit_threshold=int(os.getenv("GENERATE_PROVIDER_CIRCUIT_FAILS", "3")),
            circuit_cooldown_s=float(os.getenv("GENERATE_PROVIDER_CIRCUIT_COOLDOWN_S", "30")),
            auth_cooldown_s=float(os.getenv("GENERATE_PROVIDER_AUTH_COOLDOWN_S", "600")),
            rate_limit_cooldown_s=float(
                os.getenv("GENERATE_PROVIDER_RATE_LIMIT_COOLDOWN_S", "120")
            ),
        )
    return _provider_state_manager


def reset_state_manager() -> None:
    """Reset the global state manager (useful for testing)"""
    global _provider_state_manager
    _provider_state_manager = None
