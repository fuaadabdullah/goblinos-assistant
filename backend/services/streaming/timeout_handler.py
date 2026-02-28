"""Timeout handling for streaming requests."""

import time


class StreamTimeoutHandler:
    """Handler for streaming timeouts."""

    def __init__(self, timeout_seconds: int = 300):
        """Initialize with timeout configuration."""
        self.timeout_seconds = timeout_seconds

    async def check_timeout(self, start_time: float) -> bool:
        """Check if request has timed out."""
        return time.time() - start_time > self.timeout_seconds
