"""
ChatHandler Service for managing chat completion workflows.

This service handles the complex chat completion logic that was previously
contained in the chat_controller.py file, separating it into focused components.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
from fastapi import HTTPException
from sqlalchemy.orm import Session

from config import settings
try:
    from models import ChatMessage, ChatSession, Provider
except ImportError:  # Minimal stubs for test environments where ORM models aren't available
    class ChatMessage:
        pass

    class ChatSession:
        pass

    class Provider:
        pass
from .utils import get_provider_config, get_provider_class, get_provider_model_config
from .chat_validator import ChatValidator
from .chat_response_builder import ChatResponseBuilder
from .chat_error_handler import ChatErrorHandler
from .chat_rate_limiter import ChatRateLimiter
from .chat_session_manager import ChatSessionManager
from .chat_provider_selector import ChatProviderSelector
from .chat_metrics_collector import ChatMetricsCollector
from .chat_cache_manager import ChatCacheManager
from .chat_timeout_handler import ChatTimeoutHandler
from .chat_retry_handler import ChatRetryHandler
from .chat_compression_handler import ChatCompressionHandler
from .chat_response_formatter import ChatResponseFormatter
from .chat_error_formatter import ChatErrorFormatter
from . import chat_handler_helpers

logger = logging.getLogger(__name__)


class ChatHandler:
    """Service for handling chat completion workflows."""

    def __init__(self, db: Session):
        """Initialize the ChatHandler."""
        self.db = db
        self.validator = ChatValidator()
        self.response_builder = ChatResponseBuilder()
        self.error_handler = ChatErrorHandler()
        self.rate_limiter = ChatRateLimiter()
        self.session_manager = ChatSessionManager()
        self.provider_selector = ChatProviderSelector()
        self.metrics_collector = ChatMetricsCollector()
        self.cache_manager = ChatCacheManager()
        self.timeout_handler = ChatTimeoutHandler()
        self.retry_handler = ChatRetryHandler()
        self.compression_handler = ChatCompressionHandler()
        self.response_formatter = ChatResponseFormatter()
        self.error_formatter = ChatErrorFormatter()

    async def handle_chat_completion(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        user_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle a chat completion request with comprehensive error handling and validation.

        Args:
            session_id: Chat session ID
            messages: List of chat messages
            model: Model to use for generation
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream responses
            user_id: User ID for rate limiting
            client_ip: Client IP for rate limiting
            request_id: Request ID for tracking

        Returns:
            Chat completion response
        """
        # Initialize chat state
        chat_state = ChatState(
            session_id=session_id,
            request_id=request_id or self._generate_request_id(),
            user_id=user_id,
            client_ip=client_ip,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            start_time=time.time(),
        )

        try:
            # Validate request and check rate limits
            await chat_handler_helpers.validate_and_check_limits(
                self.validator,
                self.rate_limiter,
                self.error_formatter,
                messages,
                model,
                temperature,
                max_tokens,
                stream,
                user_id,
                client_ip,
                session_id,
            )

            # Get or create chat session
            chat_session = await self.session_manager.get_or_create_session(
                session_id, user_id
            )

            # Get provider configuration
            provider_config = await self.provider_selector.select_provider(
                model, messages, chat_state
            )

            if not provider_config:
                raise HTTPException(
                    status_code=503,
                    detail=self.error_formatter.format_provider_error(
                        "No suitable provider available"
                    ),
                )

            # Initialize metrics
            self.metrics_collector.start_request(chat_state.request_id)

            # Process chat completion
            if stream:
                return await self._handle_streaming_chat(
                    chat_state, provider_config, messages
                )
            else:
                return await self._handle_non_streaming_chat(
                    chat_state, provider_config, messages
                )

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            error_response = self.error_handler.handle_chat_error(
                e, chat_state.request_id
            )
            return error_response
        finally:
            # Cleanup resources
            await self._cleanup_chat_resources(chat_state)

    async def _handle_streaming_chat(
        self,
        chat_state: ChatState,
        provider_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming chat completion."""
        provider_instance = chat_handler_helpers.resolve_provider_instance(
            provider_config
        )

        # Prepare request
        request_data = {
            "messages": messages,
            "model": chat_state.model,
            "temperature": chat_state.temperature,
            "max_tokens": chat_state.max_tokens,
            "stream": True,
        }

        try:
            # Make streaming request
            async for chunk in provider_instance.stream_chat_completion(request_data):
                # Process chunk
                processed_chunk = await self._process_chat_chunk(chunk, chat_state)

                if processed_chunk:
                    yield processed_chunk

        except Exception as e:
            logger.error(f"Provider streaming error: {e}")
            raise

    async def _handle_non_streaming_chat(
        self,
        chat_state: ChatState,
        provider_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Handle non-streaming chat completion."""
        provider_instance = chat_handler_helpers.resolve_provider_instance(
            provider_config
        )

        # Prepare request
        request_data = {
            "messages": messages,
            "model": chat_state.model,
            "temperature": chat_state.temperature,
            "max_tokens": chat_state.max_tokens,
            "stream": False,
        }

        try:
            # Make non-streaming request
            response = await provider_instance.chat_completion(request_data)

            # Process response
            processed_response = await self._process_chat_response(response, chat_state)

            return processed_response

        except Exception as e:
            logger.error(f"Provider non-streaming error: {e}")
            raise

    async def _process_chat_chunk(
        self, chunk: Dict[str, Any], chat_state: ChatState
    ) -> Optional[Dict[str, Any]]:
        """Process individual chat chunk."""
        try:
            # Validate chunk
            if not self.validator.validate_chat_chunk(chunk):
                logger.warning(f"Invalid chat chunk: {chunk}")
                return None

            # Update metrics
            self.metrics_collector.update_chunk_metrics(chat_state.request_id, chunk)

            # Format response
            formatted_response = self.response_formatter.format_chat_chunk(
                chunk, chat_state
            )

            # Apply compression if enabled
            if settings.enable_chat_compression:
                formatted_response = self.compression_handler.compress_chunk(
                    formatted_response
                )

            return formatted_response

        except Exception as e:
            logger.error(f"Error processing chat chunk: {e}")
            return None

    async def _process_chat_response(
        self, response: Dict[str, Any], chat_state: ChatState
    ) -> Dict[str, Any]:
        """Process chat completion response."""
        try:
            # Validate response
            if not self.validator.validate_chat_response(response):
                logger.warning(f"Invalid chat response: {response}")
                raise HTTPException(
                    status_code=500, detail="Invalid response from provider"
                )

            # Update metrics
            self.metrics_collector.update_response_metrics(
                chat_state.request_id, response
            )

            # Format response
            formatted_response = self.response_formatter.format_chat_response(
                response, chat_state
            )

            # Apply compression if enabled
            if settings.enable_chat_compression:
                formatted_response = self.compression_handler.compress_response(
                    formatted_response
                )

            return formatted_response

        except Exception as e:
            logger.error(f"Error processing chat response: {e}")
            raise

    async def _cleanup_chat_resources(self, chat_state: ChatState) -> None:
        """Clean up chat resources."""
        try:
            # Update session metrics
            await self.session_manager.update_session_metrics(
                chat_state.session_id,
                self.metrics_collector.get_request_metrics(chat_state.request_id),
            )

            # Clean up cache
            await self.cache_manager.cleanup_session_cache(chat_state.session_id)

            # Record metrics
            self.metrics_collector.end_request(chat_state.request_id)

        except Exception as e:
            logger.error(f"Error cleaning up chat resources: {e}")

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        import uuid

        return f"chat_{uuid.uuid4().hex[:16]}"


class ChatState:
    """State object for chat requests."""

    def __init__(
        self,
        session_id: str,
        request_id: str,
        user_id: Optional[str],
        client_ip: Optional[str],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
        start_time: float,
    ):
        self.session_id = session_id
        self.request_id = request_id
        self.user_id = user_id
        self.client_ip = client_ip
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.start_time = start_time


class ChatValidator:
    """Validator for chat requests and responses."""

    async def validate_chat_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
    ) -> ChatValidationResult:
        """Validate chat request parameters."""
        errors = []

        # Validate messages
        if not messages:
            errors.append("Messages cannot be empty")

        # Validate model
        if not model:
            errors.append("Model is required")

        # Validate temperature
        if not (0.0 <= temperature <= 2.0):
            errors.append("Temperature must be between 0.0 and 2.0")

        # Validate max_tokens
        if max_tokens is not None and max_tokens <= 0:
            errors.append("Max tokens must be positive")

        return ChatValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )

    def validate_chat_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Validate individual chat chunk."""
        required_fields = ["content"]
        return all(field in chunk for field in required_fields)

    def validate_chat_response(self, response: Dict[str, Any]) -> bool:
        """Validate chat completion response."""
        required_fields = ["content", "usage"]
        return all(field in response for field in required_fields)


class ChatResponseBuilder:
    """Builder for chat completion responses."""

    def build_chat_response(
        self,
        response_data: Dict[str, Any],
        chat_state: ChatState,
    ) -> Dict[str, Any]:
        """Build a chat completion response."""
        return {
            "id": chat_state.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chat_state.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_data.get("content", ""),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response_data.get("prompt_tokens", 0),
                "completion_tokens": response_data.get("completion_tokens", 0),
                "total_tokens": response_data.get("total_tokens", 0),
            },
        }

    def build_chat_chunk(
        self,
        chunk_data: Dict[str, Any],
        chat_state: ChatState,
        chunk_index: int,
    ) -> Dict[str, Any]:
        """Build a chat completion chunk."""
        return {
            "id": chat_state.request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": chat_state.model,
            "choices": [
                {
                    "index": chunk_index,
                    "delta": {
                        "content": chunk_data.get("content", ""),
                        "role": chunk_data.get("role", "assistant"),
                    },
                    "finish_reason": None,
                }
            ],
        }


class ChatErrorHandler:
    """Handler for chat errors."""

    def handle_chat_error(self, error: Exception, request_id: str) -> Dict[str, Any]:
        """Handle chat errors and return error response."""
        error_type = type(error).__name__
        error_message = str(error)

        return {
            "id": request_id,
            "object": "error",
            "error": {
                "type": error_type,
                "message": error_message,
                "code": 500,
            },
        }


class ChatRateLimiter:
    """Rate limiter for chat requests."""

    async def check_rate_limit(
        self, user_id: Optional[str], client_ip: Optional[str], session_id: str
    ) -> RateLimitResult:
        """Check rate limits for chat requests."""
        # Implementation would depend on rate limiting strategy
        # This is a placeholder for the actual rate limiting logic
        return RateLimitResult(
            allowed=True,
            retry_after=None,
        )


class ChatSessionManager:
    """Manager for chat sessions."""

    async def get_or_create_session(
        self, session_id: str, user_id: Optional[str]
    ) -> ChatSession:
        """Get or create a chat session."""
        # This would interact with the database
        # Placeholder implementation
        return ChatSession(
            id=session_id,
            user_id=user_id,
            created_at=time.time(),
            updated_at=time.time(),
        )

    async def update_session_metrics(self, session_id: str, metrics: ChatStats) -> None:
        """Update session metrics."""
        # This would update the database
        # Placeholder implementation
        pass


class ChatProviderSelector:
    """Manager for selecting and configuring providers."""

    async def select_provider(
        self, model: str, messages: List[Dict[str, Any]], chat_state: ChatState
    ) -> Optional[Dict[str, Any]]:
        """Select the best provider for a model and messages."""
        # This would implement provider selection logic
        # Placeholder implementation
        return {
            "provider_name": "openai",
            "model": model,
            "config": {},
        }


class ChatMetricsCollector:
    """Collector for chat metrics."""

    def __init__(self):
        """Initialize metrics collection."""
        self.request_metrics: Dict[str, ChatStats] = {}

    def start_request(self, request_id: str) -> None:
        """Start collecting metrics for a request."""
        self.request_metrics[request_id] = ChatStats(
            request_id=request_id,
            start_time=time.time(),
            chunks_processed=0,
            total_tokens=0,
            response_time=0.0,
        )

    def update_chunk_metrics(self, request_id: str, chunk: Dict[str, Any]) -> None:
        """Update metrics with chunk data."""
        if request_id in self.request_metrics:
            stats = self.request_metrics[request_id]
            stats.chunks_processed += 1
            stats.total_tokens += len(chunk.get("content", "").split())
            stats.response_time = time.time() - stats.start_time

    def update_response_metrics(
        self, request_id: str, response: Dict[str, Any]
    ) -> None:
        """Update metrics with response data."""
        if request_id in self.request_metrics:
            stats = self.request_metrics[request_id]
            stats.total_tokens += response.get("total_tokens", 0)
            stats.response_time = time.time() - stats.start_time

    def get_request_metrics(self, request_id: str) -> Optional[ChatStats]:
        """Get metrics for a request."""
        return self.request_metrics.get(request_id)

    def end_request(self, request_id: str) -> None:
        """End metrics collection for a request."""
        if request_id in self.request_metrics:
            stats = self.request_metrics[request_id]
            stats.response_time = time.time() - stats.start_time


class ChatCacheManager:
    """Manager for caching chat responses."""

    def __init__(self, cache_ttl: int = 300):
        """Initialize cache manager."""
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}

    async def get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Get cached response if not expired."""
        if cache_key in self.cache:
            response, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return response
        return None

    async def cache_response(self, cache_key: str, response: Any) -> None:
        """Cache a response."""
        self.cache[cache_key] = (response, time.time())

    async def cleanup_session_cache(self, session_id: str) -> None:
        """Clean up cache for a session."""
        keys_to_remove = [
            key for key in self.cache.keys() if key.startswith(session_id)
        ]
        for key in keys_to_remove:
            del self.cache[key]


class ChatTimeoutHandler:
    """Handler for chat timeouts."""

    def __init__(self, timeout_seconds: int = 300):
        """Initialize with timeout configuration."""
        self.timeout_seconds = timeout_seconds

    async def check_timeout(self, start_time: float) -> bool:
        """Check if request has timed out."""
        return time.time() - start_time > self.timeout_seconds


class ChatRetryHandler:
    """Handler for retrying failed chat requests."""

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


class ChatCompressionHandler:
    """Handler for compressing chat responses."""

    def __init__(self, compression_enabled: bool = False):
        """Initialize with compression configuration."""
        self.compression_enabled = compression_enabled

    def compress_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Compress a chat chunk if compression is enabled."""
        if not self.compression_enabled:
            return chunk

        # Implementation would depend on compression algorithm
        # This is a placeholder for the actual compression logic
        return chunk

    def compress_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Compress a chat response if compression is enabled."""
        if not self.compression_enabled:
            return response

        # Implementation would depend on compression algorithm
        # This is a placeholder for the actual compression logic
        return response


class ChatResponseFormatter:
    """Formatter for chat responses."""

    def format_chat_chunk(
        self, chunk_data: Dict[str, Any], chat_state: ChatState
    ) -> Dict[str, Any]:
        """Format a chat chunk for the client."""
        return {
            "id": chat_state.request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": chat_state.model,
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

    def format_chat_response(
        self, response_data: Dict[str, Any], chat_state: ChatState
    ) -> Dict[str, Any]:
        """Format a chat response for the client."""
        return {
            "id": chat_state.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chat_state.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_data.get("content", ""),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response_data.get("prompt_tokens", 0),
                "completion_tokens": response_data.get("completion_tokens", 0),
                "total_tokens": response_data.get("total_tokens", 0),
            },
        }


class ChatErrorFormatter:
    """Formatter for chat errors."""

    def format_validation_error(self, errors: List[str]) -> Dict[str, Any]:
        """Format validation errors."""
        return {
            "type": "validation_error",
            "message": "Request validation failed",
            "errors": errors,
        }

    def format_rate_limit_error(self, retry_after: Optional[float]) -> Dict[str, Any]:
        """Format rate limit errors."""
        error_msg = {"type": "rate_limit_exceeded", "message": "Rate limit exceeded"}
        if retry_after:
            error_msg["retry_after"] = retry_after
        return error_msg

    def format_provider_error(self, error_msg: str) -> Dict[str, Any]:
        """Format provider errors."""
        return {
            "type": "provider_error",
            "message": error_msg,
        }


# Type definitions for chat components
class ChatValidationResult:
    """Result of chat validation."""

    def __init__(self, is_valid: bool, errors: List[str]):
        self.is_valid = is_valid
        self.errors = errors


class RateLimitResult:
    """Result of rate limit check."""

    def __init__(self, allowed: bool, retry_after: Optional[float]):
        self.allowed = allowed
        self.retry_after = retry_after


class ChatStats:
    """Statistics for chat requests."""

    def __init__(
        self,
        request_id: str,
        start_time: float,
        chunks_processed: int = 0,
        total_tokens: int = 0,
        response_time: float = 0.0,
    ):
        self.request_id = request_id
        self.start_time = start_time
        self.chunks_processed = chunks_processed
        self.total_tokens = total_tokens
        self.response_time = response_time
