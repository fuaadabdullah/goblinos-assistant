"""
ChatResponseFormatter Service for formatting chat responses.

This service handles formatting for chat completion responses,
separating formatting concerns from the main chat handler.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ResponseFormat(Enum):
    """Types of response formats."""

    JSON = "json"
    TEXT = "text"
    STREAM = "stream"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class ResponseFormatConfig:
    """Configuration for response formatting."""

    default_format: ResponseFormat = ResponseFormat.JSON
    include_timestamp: bool = True
    include_request_id: bool = True
    include_session_info: bool = True
    include_provider_info: bool = True
    include_usage_info: bool = True
    pretty_print: bool = False
    max_response_length: int = 10000
    enable_format_validation: bool = True


@dataclass
class FormattedResponse:
    """Formatted response with metadata."""

    content: Union[str, Dict[str, Any]]
    format_type: ResponseFormat
    timestamp: float
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    provider_used: Optional[str] = None
    usage_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatResponseFormatter:
    """Service for formatting chat responses."""

    def __init__(self, config: Optional[ResponseFormatConfig] = None):
        """Initialize the ChatResponseFormatter."""
        self.config = config or ResponseFormatConfig()

    def format_response(
        self,
        response_data: Dict[str, Any],
        format_type: Optional[ResponseFormat] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        provider_used: Optional[str] = None,
    ) -> FormattedResponse:
        """
        Format a chat response.

        Args:
            response_data: Raw response data from provider
            format_type: Desired format type (optional)
            request_id: Request ID for tracking
            session_id: Session ID for tracking
            provider_used: Provider name used for the request

        Returns:
            Formatted response
        """
        try:
            # Determine format type
            target_format = format_type or self.config.default_format

            # Extract response content
            content = self._extract_content(response_data)

            # Format content based on type
            formatted_content = self._format_content(content, target_format)

            # Build response metadata
            metadata = self._build_metadata(
                request_id, session_id, provider_used, response_data
            )

            # Create formatted response
            formatted_response = FormattedResponse(
                content=formatted_content,
                format_type=target_format,
                timestamp=time.time(),
                request_id=request_id,
                session_id=session_id,
                provider_used=provider_used,
                usage_info=self._extract_usage_info(response_data),
                metadata=metadata,
            )

            # Validate format if enabled
            if self.config.enable_format_validation:
                self._validate_format(formatted_response, target_format)

            logger.debug(
                f"Formatted response for request {request_id} in {target_format.value} format"
            )
            return formatted_response

        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            # Return error response in default format
            return self._create_error_response(str(e), request_id, session_id)

    def format_stream_response(
        self,
        chunk_data: Dict[str, Any],
        chunk_index: int,
        total_chunks: Optional[int] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Format a streaming response chunk.

        Args:
            chunk_data: Chunk data from provider
            chunk_index: Index of current chunk
            total_chunks: Total number of chunks (optional)
            request_id: Request ID for tracking
            session_id: Session ID for tracking

        Returns:
            Formatted chunk as string
        """
        try:
            # Extract chunk content
            content = self._extract_stream_content(chunk_data)

            # Format as stream chunk
            chunk = {
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "content": content,
                "timestamp": time.time(),
                "request_id": request_id,
                "session_id": session_id,
            }

            # Add metadata if configured
            if self.config.include_timestamp:
                chunk["timestamp"] = time.time()
            if self.config.include_request_id and request_id:
                chunk["request_id"] = request_id
            if self.config.include_session_info and session_id:
                chunk["session_id"] = session_id

            # Format as JSON string for streaming
            return json.dumps(chunk, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error formatting stream response: {e}")
            return json.dumps({"error": str(e), "chunk_index": chunk_index})

    def format_error_response(
        self,
        error_message: str,
        error_type: str,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        provider_used: Optional[str] = None,
    ) -> FormattedResponse:
        """
        Format an error response.

        Args:
            error_message: Error message
            error_type: Type of error
            request_id: Request ID for tracking
            session_id: Session ID for tracking
            provider_used: Provider name used for the request

        Returns:
            Formatted error response
        """
        try:
            error_data = {
                "error": {
                    "type": error_type,
                    "message": error_message,
                    "timestamp": time.time(),
                }
            }

            if request_id:
                error_data["error"]["request_id"] = request_id
            if session_id:
                error_data["error"]["session_id"] = session_id
            if provider_used:
                error_data["error"]["provider"] = provider_used

            return self.format_response(
                error_data,
                ResponseFormat.JSON,
                request_id,
                session_id,
                provider_used,
            )

        except Exception as e:
            logger.error(f"Error formatting error response: {e}")
            return self._create_error_response(str(e), request_id, session_id)

    def format_usage_info(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format usage information.

        Args:
            usage_data: Raw usage data from provider

        Returns:
            Formatted usage information
        """
        try:
            formatted_usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
                "estimated_cost": self._calculate_estimated_cost(usage_data),
            }

            # Add provider-specific usage info
            if "provider_usage" in usage_data:
                formatted_usage["provider_usage"] = usage_data["provider_usage"]

            return formatted_usage

        except Exception as e:
            logger.error(f"Error formatting usage info: {e}")
            return {}

    def get_response_headers(self, format_type: ResponseFormat) -> Dict[str, str]:
        """
        Get HTTP headers for a response format.

        Args:
            format_type: Response format type

        Returns:
            Dictionary of HTTP headers
        """
        headers = {}

        if format_type == ResponseFormat.JSON:
            headers["Content-Type"] = "application/json"
        elif format_type == ResponseFormat.TEXT:
            headers["Content-Type"] = "text/plain"
        elif format_type == ResponseFormat.HTML:
            headers["Content-Type"] = "text/html"
        elif format_type == ResponseFormat.MARKDOWN:
            headers["Content-Type"] = "text/markdown"
        elif format_type == ResponseFormat.STREAM:
            headers["Content-Type"] = "text/event-stream"
            headers["Cache-Control"] = "no-cache"
            headers["Connection"] = "keep-alive"

        return headers

    def update_format_config(self, config: ResponseFormatConfig) -> None:
        """Update response format configuration."""
        self.config = config
        logger.info("Updated response format configuration")

    def get_format_stats(self) -> Dict[str, Any]:
        """Get formatting statistics."""
        try:
            return {
                "config": {
                    "default_format": self.config.default_format.value,
                    "include_timestamp": self.config.include_timestamp,
                    "include_request_id": self.config.include_request_id,
                    "include_session_info": self.config.include_session_info,
                    "include_provider_info": self.config.include_provider_info,
                    "include_usage_info": self.config.include_usage_info,
                    "pretty_print": self.config.pretty_print,
                    "max_response_length": self.config.max_response_length,
                    "enable_format_validation": self.config.enable_format_validation,
                },
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error getting format stats: {e}")
            return {"error": str(e)}

    def reset_stats(self) -> None:
        """Reset formatting statistics."""
        try:
            logger.info("Reset formatting statistics")

        except Exception as e:
            logger.error(f"Error resetting format stats: {e}")

    def _extract_content(
        self, response_data: Dict[str, Any]
    ) -> Union[str, Dict[str, Any]]:
        """Extract content from response data."""
        try:
            # Handle different response formats from providers
            if "choices" in response_data and response_data["choices"]:
                choice = response_data["choices"][0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                elif "text" in choice:
                    return choice["text"]

            # Handle direct content
            if "content" in response_data:
                return response_data["content"]

            # Handle text responses
            if "text" in response_data:
                return response_data["text"]

            # Return raw response if no specific content found
            return response_data

        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return str(response_data)

    def _format_content(
        self, content: Union[str, Dict[str, Any]], format_type: ResponseFormat
    ) -> Union[str, Dict[str, Any]]:
        """Format content based on format type."""
        try:
            if format_type == ResponseFormat.JSON:
                if isinstance(content, str):
                    # Try to parse as JSON if it's a string
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass

                if self.config.pretty_print:
                    return json.dumps(content, indent=2, ensure_ascii=False)
                else:
                    return json.dumps(content, ensure_ascii=False)

            elif format_type == ResponseFormat.TEXT:
                if isinstance(content, dict):
                    return json.dumps(content, ensure_ascii=False)
                return str(content)

            elif format_type == ResponseFormat.HTML:
                return self._format_as_html(content)

            elif format_type == ResponseFormat.MARKDOWN:
                return self._format_as_markdown(content)

            elif format_type == ResponseFormat.STREAM:
                return str(content)

            else:
                return content

        except Exception as e:
            logger.error(f"Error formatting content: {e}")
            return str(content)

    def _format_as_html(self, content: Union[str, Dict[str, Any]]) -> str:
        """Format content as HTML."""
        try:
            if isinstance(content, dict):
                content = json.dumps(content, indent=2, ensure_ascii=False)

            # Basic HTML formatting
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Chat Response</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    pre {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h2>Chat Response</h2>
                <pre>{content}</pre>
            </body>
            </html>
            """

            return html_content

        except Exception as e:
            logger.error(f"Error formatting as HTML: {e}")
            return str(content)

    def _format_as_markdown(self, content: Union[str, Dict[str, Any]]) -> str:
        """Format content as Markdown."""
        try:
            if isinstance(content, dict):
                content = json.dumps(content, indent=2, ensure_ascii=False)

            # Basic Markdown formatting
            markdown_content = f"""
# Chat Response

```
{content}
```

*Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

            return markdown_content

        except Exception as e:
            logger.error(f"Error formatting as Markdown: {e}")
            return str(content)

    def _extract_stream_content(self, chunk_data: Dict[str, Any]) -> str:
        """Extract content from stream chunk data."""
        try:
            # Handle different stream formats from providers
            if "choices" in chunk_data and chunk_data["choices"]:
                choice = chunk_data["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    return choice["delta"]["content"] or ""
                elif "text" in choice:
                    return choice["text"] or ""

            # Handle direct content
            if "content" in chunk_data:
                return chunk_data["content"] or ""

            # Handle text responses
            if "text" in chunk_data:
                return chunk_data["text"] or ""

            return ""

        except Exception as e:
            logger.error(f"Error extracting stream content: {e}")
            return ""

    def _build_metadata(
        self,
        request_id: Optional[str],
        session_id: Optional[str],
        provider_used: Optional[str],
        response_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build response metadata."""
        metadata = {}

        if self.config.include_timestamp:
            metadata["timestamp"] = time.time()

        if self.config.include_request_id and request_id:
            metadata["request_id"] = request_id

        if self.config.include_session_info and session_id:
            metadata["session_id"] = session_id

        if self.config.include_provider_info and provider_used:
            metadata["provider"] = provider_used

        # Add any additional metadata from response
        if "metadata" in response_data:
            metadata.update(response_data["metadata"])

        return metadata

    def _extract_usage_info(
        self, response_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract usage information from response."""
        try:
            if "usage" in response_data:
                return self.format_usage_info(response_data["usage"])
            return None

        except Exception as e:
            logger.error(f"Error extracting usage info: {e}")
            return None

    def _calculate_estimated_cost(self, usage_data: Dict[str, Any]) -> float:
        """Calculate estimated cost from usage data."""
        try:
            # Simple cost estimation - in a real implementation, this would use
            # actual pricing from providers
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)

            # Example pricing: $0.02 per 1000 tokens
            cost_per_1000_tokens = 0.02
            total_tokens = prompt_tokens + completion_tokens

            return (total_tokens / 1000) * cost_per_1000_tokens

        except Exception as e:
            logger.error(f"Error calculating estimated cost: {e}")
            return 0.0

    def _validate_format(
        self, formatted_response: FormattedResponse, target_format: ResponseFormat
    ) -> bool:
        """Validate that the formatted response matches the target format."""
        try:
            if target_format == ResponseFormat.JSON:
                if isinstance(formatted_response.content, str):
                    # Try to parse as JSON
                    json.loads(formatted_response.content)

            elif target_format == ResponseFormat.TEXT:
                if not isinstance(formatted_response.content, str):
                    raise ValueError("Text format must have string content")

            elif target_format == ResponseFormat.HTML:
                if not isinstance(formatted_response.content, str):
                    raise ValueError("HTML format must have string content")
                if "<html>" not in formatted_response.content.lower():
                    logger.warning("HTML content may not be properly formatted")

            elif target_format == ResponseFormat.MARKDOWN:
                if not isinstance(formatted_response.content, str):
                    raise ValueError("Markdown format must have string content")

            return True

        except Exception as e:
            logger.error(f"Format validation failed: {e}")
            return False

    def _create_error_response(
        self,
        error_message: str,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> FormattedResponse:
        """Create an error response."""
        error_data = {
            "error": {
                "type": "formatting_error",
                "message": error_message,
                "timestamp": time.time(),
            }
        }

        if request_id:
            error_data["error"]["request_id"] = request_id
        if session_id:
            error_data["error"]["session_id"] = session_id

        return FormattedResponse(
            content=json.dumps(error_data, ensure_ascii=False),
            format_type=self.config.default_format,
            timestamp=time.time(),
            request_id=request_id,
            session_id=session_id,
            metadata={"error": True},
        )

    def get_format_recommendations(self) -> List[str]:
        """Get recommendations for response formatting."""
        try:
            recommendations = []

            # Check if pretty print is enabled for JSON
            if (
                self.config.default_format == ResponseFormat.JSON
                and not self.config.pretty_print
            ):
                recommendations.append(
                    "Consider enabling pretty_print for better JSON readability"
                )

            # Check if timestamp is disabled
            if not self.config.include_timestamp:
                recommendations.append(
                    "Consider enabling include_timestamp for better debugging"
                )

            # Check if request ID tracking is disabled
            if not self.config.include_request_id:
                recommendations.append(
                    "Consider enabling include_request_id for better request tracking"
                )

            # No recommendations
            if not recommendations:
                recommendations.append(
                    "Response formatting configuration appears optimal"
                )

            return recommendations

        except Exception as e:
            logger.error(f"Error getting format recommendations: {e}")
            return ["Error calculating recommendations"]

    def get_format_health(self) -> Dict[str, Any]:
        """Get response formatting health information."""
        try:
            current_time = time.time()

            return {
                "status": "healthy",
                "config": self.get_format_stats(),
                "timestamp": current_time,
            }

        except Exception as e:
            logger.error(f"Error getting format health: {e}")
            return {"status": "error", "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """
        Get response formatter statistics.

        Returns:
            Dict containing formatting statistics
        """
        return {
            "total_responses_formatted": getattr(self, "total_responses_formatted", 0),
            "format_types_used": getattr(self, "format_types_used", {}),
            "formatting_errors": getattr(self, "formatting_errors", 0),
            "avg_formatting_time": getattr(self, "avg_formatting_time", 0.0),
            "timestamp": time.time(),
        }
