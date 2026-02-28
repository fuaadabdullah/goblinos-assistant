"""Error handling for streaming responses."""

from ..stream_utils import StreamError


class StreamErrorHandler:
    """Handler for streaming errors."""

    def handle_stream_error(self, error: Exception, request_id: str) -> StreamError:
        """Handle streaming errors and return error response."""
        error_type = type(error).__name__
        error_message = str(error)

        return StreamError(
            id=request_id,
            object="error",
            error={
                "type": error_type,
                "message": error_message,
                "code": 500,
            },
        )
