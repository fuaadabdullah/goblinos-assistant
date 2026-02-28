"""Formatter for streaming responses."""

import time
from typing import Any

from ..stream_utils import StreamState


class StreamResponseFormatter:
    """Formatter for streaming responses."""

    def format_stream_chunk(
        self, chunk_data: dict[str, Any], stream_state: StreamState
    ) -> dict[str, Any]:
        """Format a stream chunk for the client."""
        return {
            "id": stream_state.request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": stream_state.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk_data.get("content", ""),
                        "role": chunk_data.get("role", "assistant"),
                    },
                    "finish_reason": None,
                }
            ],
        }
