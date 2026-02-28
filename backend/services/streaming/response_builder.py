"""Builder for streaming responses."""

import time
from typing import Any

from ..stream_utils import (
    StreamChoice,
    StreamCompletion,
    StreamDelta,
    StreamFinishReason,
    StreamMessage,
    StreamState,
    StreamUsage,
    StreamChunk,
)


class StreamResponseBuilder:
    """Builder for streaming responses."""

    def build_stream_chunk(
        self,
        chunk_data: dict[str, Any],
        stream_state: StreamState,
        chunk_index: int,
    ) -> StreamChunk:
        """Build a stream chunk from provider data."""
        return StreamChunk(
            id=stream_state.request_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=stream_state.model,
            choices=[
                StreamChoice(
                    index=chunk_index,
                    delta=StreamDelta(
                        content=chunk_data.get("content", ""),
                        role=chunk_data.get("role", "assistant"),
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,  # Usage is only in final chunk
        )

    def build_stream_completion(
        self,
        final_data: dict[str, Any],
        stream_state: StreamState,
        total_chunks: int,
    ) -> StreamCompletion:
        """Build the final stream completion."""
        return StreamCompletion(
            id=stream_state.request_id,
            object="chat.completion",
            created=int(time.time()),
            model=stream_state.model,
            choices=[
                StreamChoice(
                    index=0,
                    message=StreamMessage(
                        role="assistant",
                        content=final_data.get("content", ""),
                    ),
                    finish_reason=StreamFinishReason.STOP,
                )
            ],
            usage=StreamUsage(
                prompt_tokens=final_data.get("prompt_tokens", 0),
                completion_tokens=final_data.get("completion_tokens", 0),
                total_tokens=final_data.get("total_tokens", 0),
            ),
        )
