"""
ChatController Service for handling chat completion requests.

This service orchestrates the entire chat completion flow,
separating orchestration concerns from the main chat handler.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager

# Import all the service classes
from .chat_validator import ChatValidator
from .chat_response_builder import ChatResponseBuilder
from .chat_error_handler import ChatErrorHandler
from .chat_rate_limiter import ChatRateLimiter
from .chat_session_manager import ChatSessionManager
from .chat_provider_selector import ChatProviderSelector
from .chat_metrics_collector import ChatMetricsCollector
from .chat_cache_manager import ChatCacheManager
from .chat_timeout_handler import ChatTimeoutHandler, TimeoutType
from .chat_retry_handler import ChatRetryHandler
from .chat_compression_handler import ChatCompressionHandler
from .chat_response_formatter import ChatResponseFormatter
from .chat_error_formatter import ChatErrorFormatter

logger = logging.getLogger(__name__)


class ChatState(Enum):
    """States of a chat request."""
    INITIALIZING = "initializing"
    VALIDATING = "validating"
    RATE_LIMITING = "rate_limiting"
    PROVIDER_SELECTION = "provider_selection"
    PROCESSING = "processing"
    STREAMING = "streaming"
    COMPLETING = "completing"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class ChatRequest:
    """Chat request data."""
    session_id: str
    messages: List[Dict[str, Any]]
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    user_id: Optional[str] = None
    client_ip: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponse:
    """Chat response data."""
    content: Union[str, Dict[str, Any]]
    usage: Optional[Dict[str, Any]] = None
    provider_used: Optional[str] = None
    response_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatController:
    """Service for handling chat completion requests."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        provider_selector: Any | None = None,
    ):
        """Initialize the ChatController."""
        self.config = config or {}
        
        # Initialize services
        self.validator = ChatValidator()
        self.rate_limiter = ChatRateLimiter()
        self.session_manager = ChatSessionManager()
        self.provider_selector = provider_selector or ChatProviderSelector([])
        self.response_builder = ChatResponseBuilder()
        self.error_handler = ChatErrorHandler()
        self.metrics_collector = ChatMetricsCollector()
        self.cache_manager = ChatCacheManager()
        self.timeout_handler = ChatTimeoutHandler()
        self.retry_handler = ChatRetryHandler()
        self.compression_handler = ChatCompressionHandler()
        self.response_formatter = ChatResponseFormatter()
        self.error_formatter = ChatErrorFormatter()
        
        # Request tracking
        self.active_requests: Dict[str, ChatState] = {}
        
        logger.info("ChatController initialized with all services")

    async def handle_chat_request(
        self,
        request: ChatRequest,
        request_id: str,
    ) -> Union[ChatResponse, AsyncGenerator[Dict[str, Any], None]]:
        """
        Handle a chat completion request.
        
        Args:
            request: Chat request data
            request_id: Unique request ID
            
        Returns:
            Chat response or async generator for streaming
        """
        try:
            # Set initial state
            self.active_requests[request_id] = ChatState.INITIALIZING
            
            # Start metrics collection
            self.metrics_collector.start_request(request_id, {
                "session_id": request.session_id,
                "user_id": request.user_id,
                "client_ip": request.client_ip,
                "model": request.model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": request.stream,
            })
            
            # Validate request
            self.active_requests[request_id] = ChatState.VALIDATING
            validation_result = await self.validator.validate_request(request)
            if not validation_result.is_valid:
                return await self._handle_validation_error(request_id, validation_result)
            
            # Check rate limits
            self.active_requests[request_id] = ChatState.RATE_LIMITING
            rate_limit_result = await self.rate_limiter.check_rate_limit(
                request.user_id or request.client_ip,
                request.model,
                request_id,
            )
            if not rate_limit_result.allowed:
                return await self._handle_rate_limit_error(request_id, rate_limit_result)
            
            # Get or create session
            self.active_requests[request_id] = ChatState.PROCESSING
            session = await self.session_manager.get_or_create_session(
                request.session_id,
                request.user_id,
                request.client_ip,
            )
            
            # Update session with request
            await self.session_manager.update_session_with_request(
                request.session_id,
                request.messages,
                request.model,
                request.temperature,
                request.max_tokens,
            )
            
            # Select provider
            self.active_requests[request_id] = ChatState.PROVIDER_SELECTION
            provider = await self.provider_selector.select_provider(
                request.model,
                request.messages,
                session,
            )
            if not provider:
                return await self._handle_provider_error(request_id, "No suitable provider available")
            
            # Build request for provider
            provider_request = await self.response_builder.build_provider_request(
                request,
                session,
                provider,
            )
            
            # Handle the request based on streaming preference
            if request.stream:
                return self._handle_streaming_request(
                    request_id,
                    request,
                    provider,
                    provider_request,
                )
            else:
                return await self._handle_non_streaming_request(
                    request_id,
                    request,
                    provider,
                    provider_request,
                )
                
        except Exception as e:
            logger.error(f"Error handling chat request {request_id}: {e}")
            return await self._handle_system_error(request_id, e)

    async def _handle_streaming_request(
        self,
        request_id: str,
        request: ChatRequest,
        provider: Any,
        provider_request: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle a streaming chat request."""
        try:
            self.active_requests[request_id] = ChatState.STREAMING
            
            # Use timeout handler for streaming
            async with self.timeout_handler.timeout_context(
                request_id,
                TimeoutType.STREAMING,
                self.config.get("streaming_timeout_seconds", 600),
            ):
                # Get provider client
                provider_client = await self._get_provider_client(provider)
                
                # Make streaming request with retry logic
                async for chunk in self.retry_handler.retry_with_strategy(
                    provider_client.stream_chat_completion,
                    provider_request,
                    request_id=request_id,
                ):
                    # Update metrics with chunk
                    self.metrics_collector.update_chunk_metrics(request_id, chunk)
                    
                    # Format chunk
                    formatted_chunk = self.response_formatter.format_stream_response(
                        chunk,
                        chunk_index=0,  # Would need to track this properly
                        request_id=request_id,
                        session_id=request.session_id,
                    )
                    
                    yield formatted_chunk
            
            # Complete request
            self.active_requests[request_id] = ChatState.COMPLETING
            await self._complete_request(request_id, request, provider)
            
        except Exception as e:
            logger.error(f"Error in streaming request {request_id}: {e}")
            await self._handle_streaming_error(request_id, e)
            raise

    async def _handle_non_streaming_request(
        self,
        request_id: str,
        request: ChatRequest,
        provider: Any,
        provider_request: Dict[str, Any],
    ) -> ChatResponse:
        """Handle a non-streaming chat request."""
        try:
            self.active_requests[request_id] = ChatState.PROCESSING
            
            # Use timeout handler for processing
            async with self.timeout_handler.timeout_context(
                request_id,
                TimeoutType.REQUEST,
                self.config.get("request_timeout_seconds", 300),
            ):
                # Get provider client
                provider_client = await self._get_provider_client(provider)
                
                # Make request with retry logic
                response = await self.retry_handler.retry_with_strategy(
                    provider_client.chat_completion,
                    provider_request,
                    request_id=request_id,
                )
            
            # Update metrics with response
            self.metrics_collector.update_response_metrics(request_id, response, provider.name)
            
            # Format response
            formatted_response = self.response_formatter.format_response(
                response,
                request_id=request_id,
                session_id=request.session_id,
                provider_used=provider.name,
            )
            
            # Complete request
            self.active_requests[request_id] = ChatState.COMPLETING
            await self._complete_request(request_id, request, provider)
            
            return ChatResponse(
                content=formatted_response.content,
                usage=self.metrics_collector.get_request_metrics(request_id).usage_info,
                provider_used=provider.name,
                response_time=time.time() - self.metrics_collector.get_request_metrics(request_id).start_time,
                metadata=formatted_response.metadata,
            )
            
        except Exception as e:
            logger.error(f"Error in non-streaming request {request_id}: {e}")
            await self._handle_processing_error(request_id, e)
            raise

    async def _handle_validation_error(
        self,
        request_id: str,
        validation_result: Any,
    ) -> ChatResponse:
        """Handle validation error."""
        try:
            # Format error
            formatted_error = self.error_formatter.format_validation_error(
                validation_result.errors,
                request_id,
                None,  # session_id not available yet
            )
            
            # Update metrics with error
            self.metrics_collector.update_error_metrics(
                request_id,
                "validation_error",
                formatted_error.error_message,
            )
            
            # Complete request
            await self._complete_request(request_id, None, None)
            
            return ChatResponse(
                content={"error": formatted_error.error_message},
                metadata={"error_details": formatted_error},
            )
            
        except Exception as e:
            logger.error(f"Error handling validation error for request {request_id}: {e}")
            return await self._handle_system_error(request_id, e)

    async def _handle_rate_limit_error(
        self,
        request_id: str,
        rate_limit_result: Any,
    ) -> ChatResponse:
        """Handle rate limit error."""
        try:
            # Format error
            formatted_error = self.error_formatter.format_rate_limit_error(
                rate_limit_result.limit_type,
                rate_limit_result.limit_value,
                request_id,
                None,  # session_id not available yet
            )
            
            # Update metrics with error
            self.metrics_collector.update_error_metrics(
                request_id,
                "rate_limit_error",
                formatted_error.error_message,
            )
            
            # Complete request
            await self._complete_request(request_id, None, None)
            
            return ChatResponse(
                content={"error": formatted_error.error_message},
                metadata={"error_details": formatted_error},
            )
            
        except Exception as e:
            logger.error(f"Error handling rate limit error for request {request_id}: {e}")
            return await self._handle_system_error(request_id, e)

    async def _handle_provider_error(
        self,
        request_id: str,
        error_message: str,
    ) -> ChatResponse:
        """Handle provider selection error."""
        try:
            # Format error
            formatted_error = self.error_formatter.format_error(
                ValueError(error_message),
                request_id,
                None,  # session_id not available yet
            )
            
            # Update metrics with error
            self.metrics_collector.update_error_metrics(
                request_id,
                "provider_error",
                formatted_error.error_message,
            )
            
            # Complete request
            await self._complete_request(request_id, None, None)
            
            return ChatResponse(
                content={"error": formatted_error.error_message},
                metadata={"error_details": formatted_error},
            )
            
        except Exception as e:
            logger.error(f"Error handling provider error for request {request_id}: {e}")
            return await self._handle_system_error(request_id, e)

    async def _handle_processing_error(
        self,
        request_id: str,
        error: Exception,
    ) -> ChatResponse:
        """Handle processing error."""
        try:
            # Format error
            formatted_error = self.error_formatter.format_error(
                error,
                request_id,
                None,  # session_id not available yet
            )
            
            # Update metrics with error
            self.metrics_collector.update_error_metrics(
                request_id,
                type(error).__name__,
                formatted_error.error_message,
            )
            
            # Complete request
            await self._complete_request(request_id, None, None)
            
            return ChatResponse(
                content={"error": formatted_error.error_message},
                metadata={"error_details": formatted_error},
            )
            
        except Exception as e:
            logger.error(f"Error handling processing error for request {request_id}: {e}")
            return await self._handle_system_error(request_id, e)

    async def _handle_streaming_error(
        self,
        request_id: str,
        error: Exception,
    ) -> None:
        """Handle streaming error."""
        try:
            # Format error
            formatted_error = self.error_formatter.format_error(
                error,
                request_id,
                None,  # session_id not available yet
            )
            
            # Update metrics with error
            self.metrics_collector.update_error_metrics(
                request_id,
                type(error).__name__,
                formatted_error.error_message,
            )
            
            # Complete request
            await self._complete_request(request_id, None, None)
            
        except Exception as e:
            logger.error(f"Error handling streaming error for request {request_id}: {e}")

    async def _handle_system_error(
        self,
        request_id: str,
        error: Exception,
    ) -> ChatResponse:
        """Handle system error."""
        try:
            # Format error
            formatted_error = self.error_formatter.format_error(
                error,
                request_id,
                None,  # session_id not available yet
            )
            
            # Update metrics with error
            self.metrics_collector.update_error_metrics(
                request_id,
                type(error).__name__,
                formatted_error.error_message,
            )
            
            # Complete request
            await self._complete_request(request_id, None, None)
            
            return ChatResponse(
                content={"error": "A system error occurred"},
                metadata={"error_details": formatted_error},
            )
            
        except Exception as e:
            logger.error(f"Error handling system error for request {request_id}: {e}")
            return ChatResponse(
                content={"error": "A critical system error occurred"},
                metadata={"original_error": str(e)},
            )

    async def _complete_request(
        self,
        request_id: str,
        request: Optional[ChatRequest],
        provider: Optional[Any],
    ) -> None:
        """Complete a request."""
        try:
            # Update session metrics
            if request:
                await self.session_manager.update_session_metrics(
                    request.session_id,
                    self.metrics_collector.get_request_metrics(request_id),
                )
            
            # Cleanup cache if needed
            if request:
                await self.cache_manager.cleanup_session_cache(request.session_id)
            
            # End metrics collection
            self.metrics_collector.end_request(request_id)
            
            # Update state
            self.active_requests[request_id] = ChatState.COMPLETED
            
            logger.info(f"Completed request {request_id}")
            
        except Exception as e:
            logger.error(f"Error completing request {request_id}: {e}")

    async def _get_provider_client(self, provider: Any) -> Any:
        """Get provider client instance."""
        try:
            # This would need to be implemented based on your provider system
            # For now, return the provider as-is
            return provider
            
        except Exception as e:
            logger.error(f"Error getting provider client: {e}")
            raise

    def get_request_state(self, request_id: str) -> Optional[ChatState]:
        """Get the current state of a request."""
        return self.active_requests.get(request_id)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request."""
        try:
            if request_id in self.active_requests:
                self.active_requests[request_id] = ChatState.ERROR
                logger.info(f"Cancelled request {request_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling request {request_id}: {e}")
            return False

    def get_controller_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        try:
            return {
                "active_requests": len(self.active_requests),
                "request_states": {state.value: list(self.active_requests.values()).count(state) for state in ChatState},
                "validator_stats": self.validator.get_stats(),
                "rate_limiter_stats": self.rate_limiter.get_stats(),
                "session_manager_stats": self.session_manager.get_stats(),
                "provider_selector_stats": self.provider_selector.get_stats(),
                "response_builder_stats": self.response_builder.get_stats(),
                "error_handler_stats": self.error_handler.get_stats(),
                "metrics_collector_stats": self.metrics_collector.get_stats(),
                "cache_manager_stats": self.cache_manager.get_stats(),
                "timeout_handler_stats": self.timeout_handler.get_stats(),
                "retry_handler_stats": self.retry_handler.get_stats(),
                "compression_handler_stats": self.compression_handler.get_stats(),
                "response_formatter_stats": self.response_formatter.get_stats(),
                "error_formatter_stats": self.error_formatter.get_stats(),
                "timestamp": time.time(),
            }
            
        except Exception as e:
            logger.error(f"Error getting controller stats: {e}")
            return {"error": str(e)}

    def update_controller_config(self, config: Dict[str, Any]) -> None:
        """Update controller configuration."""
        self.config.update(config)
        logger.info("Updated controller configuration")

    def reset_stats(self) -> None:
        """Reset all service statistics."""
        try:
            self.validator.reset_stats()
            self.rate_limiter.reset_stats()
            self.session_manager.reset_stats()
            self.provider_selector.reset_stats()
            self.response_builder.reset_stats()
            self.error_handler.reset_stats()
            self.metrics_collector.reset_stats()
            self.cache_manager.reset_stats()
            self.timeout_handler.reset_stats()
            self.retry_handler.reset_stats()
            self.compression_handler.reset_stats()
            self.response_formatter.reset_stats()
            self.error_formatter.reset_stats()
            
            logger.info("Reset all service statistics")
            
        except Exception as e:
            logger.error(f"Error resetting stats: {e}")

    def get_controller_health(self) -> Dict[str, Any]:
        """Get controller health information."""
        try:
            current_time = time.time()
            
            # Check for potential issues
            issues = []
            
            # Check active requests
            if len(self.active_requests) > 100:
                issues.append(f"Too many active requests: {len(self.active_requests)}")
            
            # Check for stuck requests
            stuck_requests = [
                request_id for request_id, state in self.active_requests.items()
                if state in [ChatState.ERROR, ChatState.INITIALIZING]
            ]
            if stuck_requests:
                issues.append(f"Stuck requests detected: {len(stuck_requests)}")
            
            return {
                "status": "healthy" if not issues else "warning",
                "issues": issues,
                "stats": self.get_controller_stats(),
                "timestamp": current_time,
            }
            
        except Exception as e:
            logger.error(f"Error getting controller health: {e}")
            return {"status": "error", "error": str(e)}

    def get_controller_recommendations(self) -> List[str]:
        """Get recommendations for controller configuration."""
        try:
            recommendations = []
            
            # Check active requests
            if len(self.active_requests) > 50:
                recommendations.append("Consider reducing concurrent request limits")
            
            # Check for stuck requests
            stuck_requests = [
                request_id for request_id, state in self.active_requests.items()
                if state in [ChatState.ERROR, ChatState.INITIALIZING]
            ]
            if stuck_requests:
                recommendations.append(f"Review {len(stuck_requests)} stuck requests")
            
            # No recommendations
            if not recommendations:
                recommendations.append("Controller configuration appears optimal")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting controller recommendations: {e}")
            return ["Error calculating recommendations"]
