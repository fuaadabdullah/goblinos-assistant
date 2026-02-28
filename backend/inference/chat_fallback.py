"""Chat Fallback Module

Provides a small, fast local model for chat resilience when primary
providers are unavailable or for simple queries that don't need
large models. Uses Ollama with phi3:mini or similar small models.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)


class FallbackReason(str, Enum):
    """Reason for using fallback"""
    PRIMARY_TIMEOUT = "primary_timeout"
    PRIMARY_ERROR = "primary_error"
    PRIMARY_OVERLOADED = "primary_overloaded"
    SIMPLE_QUERY = "simple_query"
    COST_OPTIMIZATION = "cost_optimization"
    EXPLICIT_REQUEST = "explicit_request"


@dataclass
class FallbackConfig:
    """Configuration for chat fallback"""
    ollama_url: str = "http://localhost:11434"
    model: str = "phi3:mini"
    backup_model: str = "tinyllama"
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: float = 30.0
    max_retries: int = 2
    retry_delay: float = 1.0
    
    # Query classification thresholds
    simple_query_max_tokens: int = 50
    simple_query_keywords: list[str] = field(default_factory=lambda: [
        "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
        "yes", "no", "ok", "okay", "sure", "help", "what time", "weather",
        "how are you", "what's up", "good morning", "good evening"
    ])


@dataclass
class FallbackResponse:
    """Response from fallback model"""
    content: str
    model: str
    reason: FallbackReason
    latency_ms: float
    tokens_used: Optional[int] = None
    fallback_chain: list[str] = field(default_factory=list)


class ChatFallback:
    """
    Chat fallback system using small local models.
    
    Provides resilient chat functionality by:
    1. Detecting simple queries that don't need large models
    2. Falling back when primary providers fail
    3. Using progressively smaller models if issues occur
    """
    
    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._model_available: dict[str, bool] = {}
        self._last_health_check: float = 0
        self._health_check_interval: float = 60.0
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.ollama_url,
                timeout=httpx.Timeout(self.config.timeout)
            )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def check_health(self) -> bool:
        """Check if Ollama is available"""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    async def ensure_model_available(self, model: str) -> bool:
        """Ensure a model is pulled and available"""
        current_time = time.time()
        
        # Use cached result if recent
        if model in self._model_available:
            if current_time - self._last_health_check < self._health_check_interval:
                return self._model_available[model]
        
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                available_models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                model_base = model.split(":")[0]
                
                self._model_available[model] = model_base in available_models
                self._last_health_check = current_time
                
                if not self._model_available[model]:
                    logger.info(f"Model {model} not found, attempting to pull...")
                    await self._pull_model(model)
                    self._model_available[model] = True
                
                return self._model_available[model]
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            self._model_available[model] = False
        
        return False
    
    async def _pull_model(self, model: str) -> None:
        """Pull a model from Ollama registry"""
        client = await self._get_client()
        
        # Pulling can take a while, use longer timeout
        async with httpx.AsyncClient(
            base_url=self.config.ollama_url,
            timeout=httpx.Timeout(600.0)  # 10 minute timeout for pulling
        ) as pull_client:
            response = await pull_client.post(
                "/api/pull",
                json={"name": model, "stream": False}
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to pull model {model}: {response.text}")
            
            logger.info(f"Successfully pulled model {model}")
    
    def is_simple_query(self, message: str) -> bool:
        """
        Detect if a query is simple enough for the fallback model.
        
        Simple queries include:
        - Greetings and farewells
        - Short yes/no questions
        - Basic conversational exchanges
        """
        message_lower = message.lower().strip()
        
        # Check word count
        word_count = len(message_lower.split())
        if word_count <= 5:
            return True
        
        # Check for simple query keywords
        for keyword in self.config.simple_query_keywords:
            if keyword in message_lower:
                return True
        
        # Check token estimate (rough: 4 chars per token)
        estimated_tokens = len(message) // 4
        if estimated_tokens <= self.config.simple_query_max_tokens:
            # Additional heuristic: no complex punctuation or code markers
            complex_markers = ["```", "def ", "class ", "function ", "{", "}", "SELECT ", "INSERT "]
            if not any(marker in message for marker in complex_markers):
                return True
        
        return False
    
    async def generate(
        self,
        messages: list[dict],
        reason: FallbackReason = FallbackReason.EXPLICIT_REQUEST,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> FallbackResponse:
        """
        Generate a response using the fallback model.
        
        Args:
            messages: Chat messages in OpenAI format
            reason: Why fallback is being used
            max_tokens: Override max tokens
            temperature: Override temperature
            
        Returns:
            FallbackResponse with generated content
        """
        start_time = time.time()
        fallback_chain = []
        
        # Try primary fallback model
        models_to_try = [self.config.model, self.config.backup_model]
        
        for model in models_to_try:
            fallback_chain.append(model)
            
            try:
                if not await self.ensure_model_available(model):
                    logger.warning(f"Model {model} not available, trying next")
                    continue
                
                response = await self._generate_with_model(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens or self.config.max_tokens,
                    temperature=temperature or self.config.temperature
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                return FallbackResponse(
                    content=response["content"],
                    model=model,
                    reason=reason,
                    latency_ms=latency_ms,
                    tokens_used=response.get("tokens"),
                    fallback_chain=fallback_chain
                )
                
            except Exception as e:
                logger.warning(f"Fallback model {model} failed: {e}")
                continue
        
        # All models failed - return error message
        latency_ms = (time.time() - start_time) * 1000
        return FallbackResponse(
            content="I apologize, but I'm having trouble responding right now. Please try again in a moment.",
            model="error",
            reason=reason,
            latency_ms=latency_ms,
            fallback_chain=fallback_chain
        )
    
    async def _generate_with_model(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float
    ) -> dict:
        """Generate response with specific model"""
        client = await self._get_client()
        
        # Convert messages to Ollama format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(
                    "/api/chat",
                    json={
                        "model": model,
                        "messages": formatted_messages,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature
                        }
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "content": data.get("message", {}).get("content", ""),
                        "tokens": data.get("eval_count")
                    }
                else:
                    logger.warning(f"Ollama returned {response.status_code}: {response.text}")
                    
            except httpx.TimeoutException:
                logger.warning(f"Timeout on attempt {attempt + 1} for model {model}")
            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1}: {e}")
            
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay)
        
        raise RuntimeError(f"Failed to generate with {model} after {self.config.max_retries} attempts")
    
    async def stream_generate(
        self,
        messages: list[dict],
        reason: FallbackReason = FallbackReason.EXPLICIT_REQUEST,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response using the fallback model.
        
        Yields chunks of the response as they're generated.
        """
        model = self.config.model
        
        # Ensure model is available
        if not await self.ensure_model_available(model):
            model = self.config.backup_model
            if not await self.ensure_model_available(model):
                yield "I apologize, but I'm having trouble responding right now."
                return
        
        client = await self._get_client()
        
        # Convert messages to Ollama format
        formatted_messages = [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in messages
        ]
        
        try:
            async with client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": formatted_messages,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens or self.config.max_tokens,
                        "temperature": temperature or self.config.temperature
                    }
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            import json
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield "I encountered an error while generating the response."
    
    async def should_use_fallback(
        self,
        message: str,
        primary_available: bool = True,
        primary_latency_ms: Optional[float] = None,
        cost_sensitive: bool = False
    ) -> tuple[bool, FallbackReason]:
        """
        Determine if fallback should be used for this query.
        
        Returns:
            Tuple of (should_use, reason)
        """
        # Primary not available
        if not primary_available:
            return True, FallbackReason.PRIMARY_ERROR
        
        # Simple query detection
        if self.is_simple_query(message):
            return True, FallbackReason.SIMPLE_QUERY
        
        # Cost optimization mode
        if cost_sensitive and len(message) < 500:
            return True, FallbackReason.COST_OPTIMIZATION
        
        # Primary is slow/overloaded
        if primary_latency_ms and primary_latency_ms > 5000:  # 5 second threshold
            return True, FallbackReason.PRIMARY_OVERLOADED
        
        return False, FallbackReason.EXPLICIT_REQUEST


# Convenience function for quick fallback
async def quick_fallback_response(
    message: str,
    system_prompt: Optional[str] = None,
    ollama_url: str = "http://localhost:11434"
) -> str:
    """
    Quick convenience function for simple fallback responses.
    
    Args:
        message: User message
        system_prompt: Optional system prompt
        ollama_url: Ollama server URL
        
    Returns:
        Generated response text
    """
    config = FallbackConfig(ollama_url=ollama_url)
    fallback = ChatFallback(config)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})
    
    try:
        response = await fallback.generate(messages)
        return response.content
    finally:
        await fallback.close()
