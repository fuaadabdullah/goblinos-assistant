"""
ResponseBuilder Service for constructing chat completion responses.

This service handles all response building logic, separating it from
the main chat completion workflow for better testability and maintainability.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from ..schemas.v1.chat import ChatCompletionResponse

logger = logging.getLogger(__name__)


class ResponseBuilder:
    """Service for building chat completion responses."""

    def __init__(self):
        """Initialize the ResponseBuilder."""
        self.response_format_version = "1.0"

    def build_response_data(
        self,
        request_id: Optional[str],
        provider_info: Optional[Dict[str, Any]],
        selected_model: str,
        response_text: str,
        routing_result: Optional[Dict[str, Any]],
        response_time_ms: float,
        tokens_used: int,
        success: bool,
        verification_result: Optional[Dict[str, Any]] = None,
        confidence_result: Optional[Dict[str, Any]] = None,
        rag_context: Optional[str] = None,
        scaling_result: Optional[Dict[str, Any]] = None,
        escalated: bool = False,
        original_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete response data structure.

        Args:
            request_id: The request ID
            provider_info: Provider information
            selected_model: The selected model
            response_text: The generated response text
            routing_result: Routing result information
            response_time_ms: Response time in milliseconds
            tokens_used: Number of tokens used
            success: Whether the request was successful
            verification_result: Verification result (optional)
            confidence_result: Confidence result (optional)
            rag_context: RAG context (optional)
            scaling_result: Scaling result (optional)
            escalated: Whether the request was escalated (optional)
            original_model: Original model before escalation (optional)

        Returns:
            Dict containing the complete response data
        """
        # Build the response structure
        response_data = {
            "id": request_id or self._generate_request_id(),
            "object": "chat.completion",
            "created": datetime.utcnow().isoformat() + "Z",
            "model": selected_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop" if success else "error",
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Will be calculated based on input
                "completion_tokens": tokens_used,
                "total_tokens": tokens_used,
            },
            "metadata": {
                "response_time_ms": response_time_ms,
                "success": success,
                "provider": provider_info,
                "routing": routing_result,
                "verification": verification_result,
                "confidence": confidence_result,
                "rag_context": rag_context,
                "scaling": scaling_result,
                "escalated": escalated,
                "original_model": original_model,
                "version": self.response_format_version,
            },
        }

        # Calculate prompt tokens if we have routing result
        if routing_result and "messages" in routing_result:
            response_data["usage"]["prompt_tokens"] = self._calculate_prompt_tokens(
                routing_result["messages"]
            )
            response_data["usage"]["total_tokens"] += response_data["usage"][
                "prompt_tokens"
            ]

        return response_data

    def build_error_response(
        self,
        error_message: str,
        error_type: str,
        request_id: Optional[str] = None,
        status_code: int = 500,
    ) -> Dict[str, Any]:
        """
        Build an error response.

        Args:
            error_message: The error message
            error_type: The type of error
            request_id: The request ID (optional)
            status_code: The HTTP status code (optional)

        Returns:
            Dict containing the error response
        """
        return {
            "error": {
                "message": error_message,
                "type": error_type,
                "param": None,
                "code": status_code,
            },
            "id": request_id or self._generate_request_id(),
            "object": "error",
            "created": datetime.utcnow().isoformat() + "Z",
            "metadata": {
                "success": False,
                "error_type": error_type,
                "status_code": status_code,
                "version": self.response_format_version,
            },
        }

    def build_stream_response(
        self,
        chunk_text: str,
        chunk_index: int,
        request_id: Optional[str] = None,
        is_final: bool = False,
    ) -> Dict[str, Any]:
        """
        Build a streaming response chunk.

        Args:
            chunk_text: The text chunk
            chunk_index: The chunk index
            request_id: The request ID
            is_final: Whether this is the final chunk

        Returns:
            Dict containing the streaming response
        """
        return {
            "id": request_id or self._generate_request_id(),
            "object": "chat.completion.chunk",
            "created": datetime.utcnow().isoformat() + "Z",
            "model": "streaming",
            "choices": [
                {
                    "index": chunk_index,
                    "delta": {"content": chunk_text},
                    "finish_reason": "stop" if is_final else None,
                }
            ],
        }

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Args:
            text: The text to estimate tokens for

        Returns:
            int: Estimated number of tokens
        """
        if not text:
            return 0

        # Rough estimation: 4 characters per token
        # This is a simplified estimation and may not be accurate for all cases
        return max(1, len(text) // 4)

    def _calculate_prompt_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Calculate the number of tokens used by the prompt messages.

        Args:
            messages: The list of messages

        Returns:
            int: Estimated number of prompt tokens
        """
        total_tokens = 0
        for message in messages:
            if "content" in message:
                total_tokens += self.estimate_tokens(message["content"])
            if "role" in message:
                total_tokens += 1  # Small overhead for role tokens

        return total_tokens

    def _generate_request_id(self) -> str:
        """
        Generate a unique request ID.

        Returns:
            str: A unique request ID
        """
        import uuid

        return f"req_{uuid.uuid4().hex[:16]}"

    def build_provider_info(
        self,
        provider_name: str,
        provider_display_name: str,
        model_name: str,
        capabilities: List[str],
        priority: int,
    ) -> Dict[str, Any]:
        """
        Build provider information structure.

        Args:
            provider_name: The provider name
            provider_display_name: The provider display name
            model_name: The model name
            capabilities: List of capabilities
            priority: Provider priority

        Returns:
            Dict containing provider information
        """
        return {
            "name": provider_name,
            "display_name": provider_display_name,
            "model": model_name,
            "capabilities": capabilities,
            "priority": priority,
            "type": "local" if "local" in provider_name.lower() else "cloud",
        }

    def build_routing_metadata(
        self,
        request_id: str,
        provider_info: Dict[str, Any],
        selected_model: str,
        routing_explanation: Optional[str] = None,
        fallback_used: bool = False,
        emergency_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Build routing metadata for the response.

        Args:
            request_id: The request ID
            provider_info: Provider information
            selected_model: The selected model
            routing_explanation: Explanation for the routing decision
            fallback_used: Whether a fallback was used
            emergency_mode: Whether emergency mode was used

        Returns:
            Dict containing routing metadata
        """
        return {
            "request_id": request_id,
            "provider": provider_info,
            "selected_model": selected_model,
            "routing_explanation": routing_explanation,
            "fallback_used": fallback_used,
            "emergency_mode": emergency_mode,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def build_verification_metadata(
        self,
        verification_passed: bool,
        confidence_score: float,
        verification_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build verification metadata for the response.

        Args:
            verification_passed: Whether verification passed
            confidence_score: The confidence score (0.0 to 1.0)
            verification_details: Additional verification details

        Returns:
            Dict containing verification metadata
        """
        return {
            "verification_passed": verification_passed,
            "confidence_score": confidence_score,
            "verification_details": verification_details or {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def build_scaling_metadata(
        self,
        scaling_applied: bool,
        scaling_factor: float,
        original_tokens: int,
        scaled_tokens: int,
        scaling_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build scaling metadata for the response.

        Args:
            scaling_applied: Whether scaling was applied
            scaling_factor: The scaling factor applied
            original_tokens: Original token count
            scaled_tokens: Scaled token count
            scaling_reason: Reason for scaling

        Returns:
            Dict containing scaling metadata
        """
        return {
            "scaling_applied": scaling_applied,
            "scaling_factor": scaling_factor,
            "original_tokens": original_tokens,
            "scaled_tokens": scaled_tokens,
            "scaling_reason": scaling_reason,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def format_response_for_api(
        self,
        response_data: Dict[str, Any],
        api_version: str = "v1",
    ) -> Dict[str, Any]:
        """
        Format the response data for the API.

        Args:
            response_data: The response data
            api_version: The API version

        Returns:
            Dict containing the formatted response
        """
        formatted_response = {
            "api_version": api_version,
            "data": response_data,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Add any API-specific formatting
        if api_version == "v1":
            # V1 API formatting
            pass
        elif api_version == "v2":
            # V2 API formatting
            pass

        return formatted_response


# Module-level convenience functions using a singleton instance
_response_builder_instance: Optional[ResponseBuilder] = None


def _get_response_builder() -> ResponseBuilder:
    """Get or create the singleton ResponseBuilder instance."""
    global _response_builder_instance
    if _response_builder_instance is None:
        _response_builder_instance = ResponseBuilder()
    return _response_builder_instance


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Module-level convenience function.

    Args:
        text: The text to estimate tokens for

    Returns:
        int: Estimated number of tokens
    """
    return _get_response_builder().estimate_tokens(text)


def build_response_data(
    request_id: Optional[str],
    provider_info: Optional[Dict[str, Any]],
    selected_model: str,
    response_text: str,
    routing_result: Optional[Dict[str, Any]],
    response_time_ms: float,
    tokens_used: int,
    success: bool,
    verification_result: Optional[Dict[str, Any]] = None,
    confidence_result: Optional[Dict[str, Any]] = None,
    rag_context: Optional[str] = None,
    scaling_result: Optional[Dict[str, Any]] = None,
    escalated: bool = False,
    original_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a complete response data structure.

    Module-level convenience function.
    """
    return _get_response_builder().build_response_data(
        request_id=request_id,
        provider_info=provider_info,
        selected_model=selected_model,
        response_text=response_text,
        routing_result=routing_result,
        response_time_ms=response_time_ms,
        tokens_used=tokens_used,
        success=success,
        verification_result=verification_result,
        confidence_result=confidence_result,
        rag_context=rag_context,
        scaling_result=scaling_result,
        escalated=escalated,
        original_model=original_model,
    )


def build_error_response(
    error_message: str,
    error_type: str,
    request_id: Optional[str] = None,
    status_code: int = 500,
) -> Dict[str, Any]:
    """
    Build an error response.

    Module-level convenience function.
    """
    return _get_response_builder().build_error_response(
        error_message=error_message,
        error_type=error_type,
        request_id=request_id,
        status_code=status_code,
    )


def build_stream_response(
    chunk_text: str,
    chunk_index: int,
    request_id: Optional[str] = None,
    is_final: bool = False,
) -> Dict[str, Any]:
    """
    Build a streaming response chunk.

    Module-level convenience function.
    """
    return _get_response_builder().build_stream_response(
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        request_id=request_id,
        is_final=is_final,
    )
