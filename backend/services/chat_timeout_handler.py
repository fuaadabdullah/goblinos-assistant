"""
ChatTimeoutHandler Service for handling chat timeouts.

This service handles timeout management for chat completion requests,
separating timeout concerns from the main chat handler.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class TimeoutType(Enum):
    """Types of timeouts."""

    REQUEST = "request"
    PROVIDER = "provider"
    STREAMING = "streaming"
    VALIDATION = "validation"


@dataclass
class TimeoutConfig:
    """Configuration for timeouts."""

    request_timeout_seconds: int = 300  # 5 minutes
    provider_timeout_seconds: int = 120  # 2 minutes
    streaming_timeout_seconds: int = 600  # 10 minutes
    validation_timeout_seconds: int = 30  # 30 seconds
    enable_timeout_monitoring: bool = True


@dataclass
class TimeoutInfo:
    """Information about a timeout."""

    request_id: str
    timeout_type: TimeoutType
    timeout_seconds: int
    start_time: float
    expected_end_time: float
    cancelled: bool = False
    error_message: Optional[str] = None


class ChatTimeoutHandler:
    """Service for handling chat timeouts."""

    def __init__(self, config: Optional[TimeoutConfig] = None):
        """Initialize the ChatTimeoutHandler."""
        self.config = config or TimeoutConfig()

        # Active timeouts tracking
        self.active_timeouts: Dict[str, TimeoutInfo] = {}

        # Timeout tasks
        self.timeout_tasks: Dict[str, asyncio.Task] = {}

        # Statistics
        self.stats = {
            "timeouts_triggered": 0,
            "timeouts_cancelled": 0,
            "timeouts_skipped": 0,
            "total_requests": 0,
        }

    @asynccontextmanager
    async def timeout_context(
        self,
        request_id: str,
        timeout_type: TimeoutType,
        custom_timeout: Optional[int] = None,
    ):
        """
        Context manager for handling timeouts.

        Args:
            request_id: Request ID
            timeout_type: Type of timeout
            custom_timeout: Custom timeout in seconds (optional)
        """
        timeout_seconds = custom_timeout or self._get_timeout_for_type(timeout_type)

        # Create timeout info
        timeout_info = TimeoutInfo(
            request_id=request_id,
            timeout_type=timeout_type,
            timeout_seconds=timeout_seconds,
            start_time=time.time(),
            expected_end_time=time.time() + timeout_seconds,
        )

        self.active_timeouts[request_id] = timeout_info
        self.stats["total_requests"] += 1

        # Start timeout monitoring task
        if self.config.enable_timeout_monitoring:
            timeout_task = asyncio.create_task(
                self._monitor_timeout(request_id, timeout_info)
            )
            self.timeout_tasks[request_id] = timeout_task

        try:
            # Use asyncio.timeout to enforce the timeout
            async with asyncio.timeout(timeout_seconds):
                yield
        except asyncio.TimeoutError as e:
            # Handle timeout
            timeout_info.cancelled = True
            timeout_info.error_message = (
                f"Request {request_id} timed out after {timeout_seconds} seconds"
            )
            self.stats["timeouts_triggered"] += 1
            logger.warning(
                f"Timeout triggered for request {request_id}: {timeout_info.error_message}"
            )
            raise
        except Exception as e:
            # Other exceptions - cancel timeout
            await self.cancel_timeout(request_id)
            raise

        finally:
            # Cleanup
            await self.cancel_timeout(request_id)
            if request_id in self.active_timeouts:
                del self.active_timeouts[request_id]
            if request_id in self.timeout_tasks:
                del self.timeout_tasks[request_id]

    async def set_timeout(
        self,
        request_id: str,
        timeout_type: TimeoutType,
        callback: callable,
        custom_timeout: Optional[int] = None,
    ) -> str:
        """
        Set a timeout with a callback.

        Args:
            request_id: Request ID
            timeout_type: Type of timeout
            callback: Callback function to call on timeout
            custom_timeout: Custom timeout in seconds (optional)

        Returns:
            Timeout ID
        """
        try:
            timeout_seconds = custom_timeout or self._get_timeout_for_type(timeout_type)

            # Create timeout info
            timeout_info = TimeoutInfo(
                request_id=request_id,
                timeout_type=timeout_type,
                timeout_seconds=timeout_seconds,
                start_time=time.time(),
                expected_end_time=time.time() + timeout_seconds,
            )

            self.active_timeouts[request_id] = timeout_info

            # Create timeout task
            timeout_task = asyncio.create_task(
                self._execute_with_timeout(request_id, timeout_info, callback)
            )
            self.timeout_tasks[request_id] = timeout_task

            logger.debug(
                f"Set timeout for request {request_id}: {timeout_seconds} seconds"
            )
            return request_id

        except Exception as e:
            logger.error(f"Error setting timeout for request {request_id}: {e}")
            return ""

    async def cancel_timeout(self, request_id: str) -> bool:
        """
        Cancel a timeout.

        Args:
            request_id: Request ID

        Returns:
            True if successfully cancelled, False if not found
        """
        try:
            if request_id in self.timeout_tasks:
                task = self.timeout_tasks[request_id]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                del self.timeout_tasks[request_id]
                self.stats["timeouts_cancelled"] += 1
                logger.debug(f"Cancelled timeout for request {request_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error cancelling timeout for request {request_id}: {e}")
            return False

    async def get_timeout_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a timeout.

        Args:
            request_id: Request ID

        Returns:
            Timeout status information, or None if not found
        """
        try:
            if request_id not in self.active_timeouts:
                return None

            timeout_info = self.active_timeouts[request_id]
            current_time = time.time()

            return {
                "request_id": timeout_info.request_id,
                "timeout_type": timeout_info.timeout_type.value,
                "timeout_seconds": timeout_info.timeout_seconds,
                "start_time": timeout_info.start_time,
                "expected_end_time": timeout_info.expected_end_time,
                "cancelled": timeout_info.cancelled,
                "error_message": timeout_info.error_message,
                "elapsed_time": current_time - timeout_info.start_time,
                "time_remaining": max(0, timeout_info.expected_end_time - current_time),
                "active": not timeout_info.cancelled
                and current_time < timeout_info.expected_end_time,
            }

        except Exception as e:
            logger.error(f"Error getting timeout status for request {request_id}: {e}")
            return None

    def get_timeout_for_type(self, timeout_type: TimeoutType) -> int:
        """Get timeout value for a specific type."""
        return self._get_timeout_for_type(timeout_type)

    def update_timeout_config(self, config: TimeoutConfig) -> None:
        """Update timeout configuration."""
        self.config = config
        logger.info("Updated timeout configuration")

    def get_stats(self) -> Dict[str, Any]:
        """Get timeout statistics."""
        try:
            current_time = time.time()

            # Calculate active timeouts
            active_count = 0
            for timeout_info in self.active_timeouts.values():
                if (
                    not timeout_info.cancelled
                    and current_time < timeout_info.expected_end_time
                ):
                    active_count += 1

            # Calculate timeout rates
            total_requests = self.stats["total_requests"]
            timeout_rate = self.stats["timeouts_triggered"] / max(total_requests, 1)
            cancellation_rate = self.stats["timeouts_cancelled"] / max(
                total_requests, 1
            )

            return {
                "active_timeouts": active_count,
                "total_requests": total_requests,
                "timeouts_triggered": self.stats["timeouts_triggered"],
                "timeouts_cancelled": self.stats["timeouts_cancelled"],
                "timeouts_skipped": self.stats["timeouts_skipped"],
                "timeout_rate": timeout_rate,
                "cancellation_rate": cancellation_rate,
                "config": {
                    "request_timeout_seconds": self.config.request_timeout_seconds,
                    "provider_timeout_seconds": self.config.provider_timeout_seconds,
                    "streaming_timeout_seconds": self.config.streaming_timeout_seconds,
                    "validation_timeout_seconds": self.config.validation_timeout_seconds,
                    "enable_timeout_monitoring": self.config.enable_timeout_monitoring,
                },
                "timestamp": current_time,
            }

        except Exception as e:
            logger.error(f"Error getting timeout stats: {e}")
            return {"error": str(e)}

    def cleanup_expired_timeouts(self) -> int:
        """
        Clean up expired timeouts.

        Returns:
            Number of timeouts cleaned up
        """
        try:
            current_time = time.time()
            expired_requests = []

            for request_id, timeout_info in self.active_timeouts.items():
                if (
                    current_time >= timeout_info.expected_end_time
                    and not timeout_info.cancelled
                ):
                    expired_requests.append(request_id)

            for request_id in expired_requests:
                # Force cancel expired timeouts
                asyncio.create_task(self.cancel_timeout(request_id))

            if expired_requests:
                logger.info(f"Cleaned up {len(expired_requests)} expired timeouts")

            return len(expired_requests)

        except Exception as e:
            logger.error(f"Error cleaning up expired timeouts: {e}")
            return 0

    def reset_stats(self) -> None:
        """Reset timeout statistics."""
        try:
            self.stats = {
                "timeouts_triggered": 0,
                "timeouts_cancelled": 0,
                "timeouts_skipped": 0,
                "total_requests": 0,
            }
            logger.info("Reset timeout statistics")

        except Exception as e:
            logger.error(f"Error resetting timeout stats: {e}")

    def get_active_timeouts(self) -> List[Dict[str, Any]]:
        """Get information about all active timeouts."""
        try:
            current_time = time.time()
            active_timeouts = []

            for request_id, timeout_info in self.active_timeouts.items():
                if (
                    not timeout_info.cancelled
                    and current_time < timeout_info.expected_end_time
                ):
                    timeout_status = self.get_timeout_status(request_id)
                    if timeout_status:
                        active_timeouts.append(timeout_status)

            return active_timeouts

        except Exception as e:
            logger.error(f"Error getting active timeouts: {e}")
            return []

    def get_timeout_health(self) -> Dict[str, Any]:
        """Get timeout health information."""
        try:
            stats = self.get_stats()
            current_time = time.time()

            # Check for potential issues
            issues = []

            # Check timeout rate
            if stats["timeout_rate"] > 0.1:  # More than 10% timeout rate
                issues.append(f"High timeout rate: {stats['timeout_rate']:.2%}")

            # Check active timeouts
            if stats["active_timeouts"] > 100:  # Too many active timeouts
                issues.append(f"Too many active timeouts: {stats['active_timeouts']}")

            # Check for old timeouts
            oldest_timeout = None
            for timeout_info in self.active_timeouts.values():
                if not timeout_info.cancelled:
                    if (
                        oldest_timeout is None
                        or timeout_info.start_time < oldest_timeout.start_time
                    ):
                        oldest_timeout = timeout_info

            if oldest_timeout:
                age = current_time - oldest_timeout.start_time
                if age > self.config.request_timeout_seconds * 2:
                    issues.append(f"Very old timeout detected: {age:.1f} seconds")

            return {
                "status": "healthy" if not issues else "warning",
                "issues": issues,
                "stats": stats,
                "timestamp": current_time,
            }

        except Exception as e:
            logger.error(f"Error getting timeout health: {e}")
            return {"status": "error", "error": str(e)}

    async def _monitor_timeout(
        self, request_id: str, timeout_info: TimeoutInfo
    ) -> None:
        """Monitor a timeout and trigger it if needed."""
        try:
            await asyncio.sleep(timeout_info.timeout_seconds)

            # Check if timeout is still active
            if request_id in self.active_timeouts and not timeout_info.cancelled:
                timeout_info.cancelled = True
                timeout_info.error_message = f"Request {request_id} timed out after {timeout_info.timeout_seconds} seconds"
                self.stats["timeouts_triggered"] += 1

                logger.warning(f"Timeout triggered for request {request_id}")

                # Cancel any associated tasks
                if request_id in self.timeout_tasks:
                    task = self.timeout_tasks[request_id]
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

        except asyncio.CancelledError:
            # Timeout was cancelled
            pass
        except Exception as e:
            logger.error(f"Error in timeout monitor for request {request_id}: {e}")

    async def _execute_with_timeout(
        self,
        request_id: str,
        timeout_info: TimeoutInfo,
        callback: callable,
    ) -> None:
        """Execute a callback with timeout monitoring."""
        try:
            await asyncio.wait_for(callback(), timeout=timeout_info.timeout_seconds)
        except asyncio.TimeoutError:
            timeout_info.cancelled = True
            timeout_info.error_message = f"Callback for request {request_id} timed out after {timeout_info.timeout_seconds} seconds"
            self.stats["timeouts_triggered"] += 1
            logger.warning(f"Callback timeout for request {request_id}")
        except Exception as e:
            logger.error(f"Error executing callback for request {request_id}: {e}")
        finally:
            # Clean up
            if request_id in self.active_timeouts:
                del self.active_timeouts[request_id]
            if request_id in self.timeout_tasks:
                del self.timeout_tasks[request_id]

    def _get_timeout_for_type(self, timeout_type: TimeoutType) -> int:
        """Get timeout value for a specific type."""
        if timeout_type == TimeoutType.REQUEST:
            return self.config.request_timeout_seconds
        elif timeout_type == TimeoutType.PROVIDER:
            return self.config.provider_timeout_seconds
        elif timeout_type == TimeoutType.STREAMING:
            return self.config.streaming_timeout_seconds
        elif timeout_type == TimeoutType.VALIDATION:
            return self.config.validation_timeout_seconds
        else:
            return self.config.request_timeout_seconds

    async def create_timeout_task(
        self,
        request_id: str,
        timeout_seconds: int,
        task_func: callable,
        *args,
        **kwargs,
    ) -> asyncio.Task:
        """
        Create a task with timeout handling.

        Args:
            request_id: Request ID
            timeout_seconds: Timeout in seconds
            task_func: Function to execute
            *args, **kwargs: Arguments for the task function

        Returns:
            Asyncio task
        """

        async def wrapped_task():
            try:
                async with asyncio.timeout(timeout_seconds):
                    return await task_func(*args, **kwargs)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Task timeout for request {request_id} after {timeout_seconds} seconds"
                )
                raise

        return asyncio.create_task(wrapped_task())

    def get_timeout_recommendations(self) -> List[str]:
        """Get recommendations for timeout configuration."""
        try:
            stats = self.get_stats()
            recommendations = []

            # High timeout rate
            if stats["timeout_rate"] > 0.05:  # More than 5%
                recommendations.append(
                    "Consider increasing timeout values due to high timeout rate"
                )

            # High cancellation rate
            if stats["cancellation_rate"] > 0.8:  # More than 80%
                recommendations.append(
                    "Consider decreasing timeout values due to high cancellation rate"
                )

            # Many active timeouts
            if stats["active_timeouts"] > 50:
                recommendations.append(
                    "Consider reducing concurrent requests or increasing timeout values"
                )

            # No recommendations
            if not recommendations:
                recommendations.append("Timeout configuration appears optimal")

            return recommendations

        except Exception as e:
            logger.error(f"Error getting timeout recommendations: {e}")
            return ["Error calculating recommendations"]
