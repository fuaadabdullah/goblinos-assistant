"""Metrics collection for streaming responses."""

import time

from ..stream_utils import StreamStats


class StreamMetricsCollector:
    """Collector for streaming metrics."""

    def __init__(self):
        """Initialize metrics collection."""
        self.request_metrics: dict[str, StreamStats] = {}

    def start_request(self, request_id: str) -> None:
        """Start collecting metrics for a request."""
        self.request_metrics[request_id] = StreamStats(
            request_id=request_id,
            start_time=time.time(),
            chunks_processed=0,
            total_tokens=0,
            response_time=0.0,
        )

    def update_chunk_metrics(self, request_id: str, chunk: dict[str, str]) -> None:
        """Update metrics with chunk data."""
        if request_id in self.request_metrics:
            stats = self.request_metrics[request_id]
            stats.chunks_processed += 1
            stats.total_tokens += len(chunk.get("content", "").split())
            stats.response_time = time.time() - stats.start_time

    def get_request_metrics(self, request_id: str) -> StreamStats | None:
        """Get metrics for a request."""
        return self.request_metrics.get(request_id)

    def end_request(self, request_id: str) -> None:
        """End metrics collection for a request."""
        if request_id in self.request_metrics:
            stats = self.request_metrics[request_id]
            stats.response_time = time.time() - stats.start_time
