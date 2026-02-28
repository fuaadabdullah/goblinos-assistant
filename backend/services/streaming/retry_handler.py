"""Retry logic for streaming requests."""

import asyncio


class StreamRetryHandler:
    """Handler for retrying failed streaming requests."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize with retry configuration."""
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(self.retry_delay * (2**attempt))
