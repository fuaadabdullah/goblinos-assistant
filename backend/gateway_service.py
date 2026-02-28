"""
Gateway service for API-level protections and enforcements.

Handles:
- Max tokens per request enforcement
- Token budget per API key enforcement
- Token usage tracking (Prometheus + DB)
- Pre-flight request classification and routing
- Anomaly detection and alerting
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import re
from .services.gateway_exceptions import (
    GatewayError,
    TokenBudgetExceeded,
    MaxTokensExceeded,
)

# Use try/except for flexible imports
try:
    from backend.anomaly_detector import get_anomaly_detector
    from backend.services.token_accounting import TokenAccountingService
except ImportError:
    from .anomaly_detector import get_anomaly_detector
    from .services.token_accounting import TokenAccountingService

logger = logging.getLogger(__name__)


class RequestIntent(Enum):
    """Classification of request intent for routing decisions."""

    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"
    RAG_SUMMARIZATION = "rag_summarization"
    CONVERSATIONAL = "conversational"
    CLASSIFICATION = "classification"
    ANALYSIS = "analysis"
    LONG_GENERATION = "long_generation"
    UNKNOWN = "unknown"


@dataclass
class GatewayResult:
    """Result of gateway processing."""

    allowed: bool
    intent: RequestIntent
    estimated_tokens: int
    risk_score: float
    recommendations: List[str]
    metadata: Dict[str, Any]


class TokenBudgetManager:
    """Manages token budgets per API key."""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        # Default budgets (configurable)
        self.default_budget = 100000  # 100k tokens per day
        self.budget_window = 86400  # 24 hours in seconds

    def _get_budget_key(self, api_key: str) -> str:
        """Get Redis key for token budget."""
        return f"gateway:budget:{api_key}"

    def _get_usage_key(self, api_key: str) -> str:
        """Get Redis key for token usage tracking."""
        return f"gateway:usage:{api_key}"

    async def check_budget(self, api_key: str, requested_tokens: int) -> bool:
        """Check if API key has budget for requested tokens."""
        if not self.redis:
            return True  # Allow if no Redis

        try:
            budget_key = self._get_budget_key(api_key)
            usage_key = self._get_usage_key(api_key)

            # Get current usage
            current_usage = await asyncio.get_event_loop().run_in_executor(
                None, lambda: int(self.redis.get(usage_key) or 0)
            )

            # Get budget limit
            budget_limit = await asyncio.get_event_loop().run_in_executor(
                None, lambda: int(self.redis.get(budget_key) or self.default_budget)
            )

            return (current_usage + requested_tokens) <= budget_limit
        except Exception as e:
            logger.warning(f"Budget check failed: {e}")
            return True  # Allow on error

    async def record_usage(self, api_key: str, tokens_used: int):
        """Record token usage for API key."""
        if not self.redis:
            return

        try:
            usage_key = self._get_usage_key(api_key)

            # Increment usage atomically
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.redis.incrby(usage_key, tokens_used)
            )

            # Set expiry on usage key (24 hours)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.redis.expire(usage_key, self.budget_window)
            )
        except Exception as e:
            logger.warning(f"Usage recording failed: {e}")

    async def get_usage_stats(self, api_key: str) -> Dict[str, Any]:
        """Get usage statistics for API key."""
        if not self.redis:
            return {"current_usage": 0, "budget_limit": self.default_budget}

        try:
            budget_key = self._get_budget_key(api_key)
            usage_key = self._get_usage_key(api_key)

            current_usage = await asyncio.get_event_loop().run_in_executor(
                None, lambda: int(self.redis.get(usage_key) or 0)
            )
            budget_limit = await asyncio.get_event_loop().run_in_executor(
                None, lambda: int(self.redis.get(budget_key) or self.default_budget)
            )

            return {
                "current_usage": current_usage,
                "budget_limit": budget_limit,
                "remaining_budget": max(0, budget_limit - current_usage),
                "usage_percentage": (current_usage / budget_limit) * 100
                if budget_limit > 0
                else 0,
            }
        except Exception as e:
            logger.warning(f"Usage stats retrieval failed: {e}")
            return {"current_usage": 0, "budget_limit": self.default_budget}


class RequestClassifier:
    """Classifies incoming requests for routing decisions."""

    def __init__(self):
        # Keywords for intent classification
        self.intent_patterns = {
            RequestIntent.CODE_GENERATION: [
                r"\b(write|create|generate|implement)\b.*\b(code|function|class|script|program)\b",
                r"\bpython|javascript|java|c\+\+|typescript|go|rust|sql\b",
                r"\bapi|endpoint|database|query\b",
                r"\bdef |function |class \b",
            ],
            RequestIntent.CREATIVE_WRITING: [
                r"\b(write|create|generate)\b.*\b(story|poem|article|blog|essay)\b",
                r"\bcreative|fiction|novel|screenplay\b",
            ],
            RequestIntent.RAG_SUMMARIZATION: [
                r"\b(summarize|summarise|tl;dr|overview)\b",
                r"\bextract|key.?points|main.?ideas\b",
                r"\bdocument|article|paper|report\b",
            ],
            RequestIntent.CLASSIFICATION: [
                r"\b(classify|categorize|label|tag)\b",
                r"\b(is|are|does|do)\b.*\b\?|what.*type\b",
            ],
            RequestIntent.ANALYSIS: [
                r"\b(analyze|analyse|review|evaluate|assess)\b",
                r"\bcompare|contrast|difference|similarity\b",
            ],
        }
        self.token_accountant = TokenAccountingService()

    def classify_request(
        self, messages: List[Dict[str, str]], context: Optional[str] = None
    ) -> RequestIntent:
        """Classify the intent of a request."""
        # Combine all message content
        content = " ".join([msg.get("content", "") for msg in messages])
        if context:
            content += " " + context

        content = content.lower()

        # Check patterns for each intent
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return intent

        # Check for long content (potential RAG or long generation)
        total_chars = len(content)
        if total_chars > 2000:  # Rough threshold
            return RequestIntent.RAG_SUMMARIZATION

        # Check for conversational patterns
        if len(messages) > 2:  # Multi-turn conversation
            return RequestIntent.CONVERSATIONAL

        return RequestIntent.UNKNOWN

    def estimate_tokens(
        self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None
    ) -> int:
        """Estimate total tokens for request + response."""
        input_tokens = sum(
            self.token_accountant.count_tokens(msg.get("content", ""))
            for msg in messages
        )

        # Estimate response size based on input
        # Typically responses are 2-3x input for conversational, less for classification
        response_multiplier = 2.5
        estimated_response_tokens = int(input_tokens * response_multiplier)

        # Add max_tokens if specified (capped)
        if max_tokens:
            estimated_response_tokens = min(estimated_response_tokens, max_tokens)

        total_tokens = input_tokens + estimated_response_tokens
        return max(1, total_tokens)


class GatewayService:
    """Main gateway service for request processing and enforcement."""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.budget_manager = TokenBudgetManager(redis_client)
        self.classifier = RequestClassifier()
        self.anomaly_detector = get_anomaly_detector(redis_client)

        # Gateway limits
        self.max_tokens_per_request = 8000  # Conservative limit
        self.high_risk_threshold = 0.7  # Risk score threshold

    async def process_request(
        self,
        messages: List[Dict[str, str]],
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> GatewayResult:
        """Process a request through gateway protections."""

        # Step 1: Classify request intent
        intent = self.classifier.classify_request(messages, context)

        # Step 2: Estimate token usage
        estimated_tokens = self.classifier.estimate_tokens(messages, max_tokens)

        # Step 3: Check max tokens limit
        if max_tokens and max_tokens > self.max_tokens_per_request:
            raise MaxTokensExceeded(
                f"max_tokens ({max_tokens}) exceeds gateway limit ({self.max_tokens_per_request})"
            )

        # Step 4: Check token budget (if API key provided)
        if api_key:
            budget_ok = await self.budget_manager.check_budget(
                api_key, estimated_tokens
            )
            if not budget_ok:
                raise TokenBudgetExceeded("Token budget exceeded for API key")

        # Step 5: Calculate risk score
        risk_score = self._calculate_risk_score(intent, estimated_tokens, max_tokens)

        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(
            intent, risk_score, estimated_tokens
        )

        # Step 7: Determine if request should be allowed
        allowed = risk_score < self.high_risk_threshold

        return GatewayResult(
            allowed=allowed,
            intent=intent,
            estimated_tokens=estimated_tokens,
            risk_score=risk_score,
            recommendations=recommendations,
            metadata={
                "api_key": api_key,
                "max_tokens_requested": max_tokens,
                "classification_confidence": "medium",  # Could be improved with ML
            },
        )

    async def record_usage(
        self,
        api_key: Optional[str],
        tokens_used: int,
        intent: Optional[RequestIntent] = None,
        success: bool = True,
        error_type: Optional[str] = None,
    ):
        """Record actual token usage after request completion."""
        if api_key:
            await self.budget_manager.record_usage(api_key, tokens_used)

            # Analyze for anomalies
            if intent:
                await self.anomaly_detector.analyze_request(
                    api_key=api_key,
                    tokens_used=tokens_used,
                    intent=intent.value,
                    success=success,
                    error_type=error_type,
                )

            # Check for budget alerts
            usage_stats = await self.budget_manager.get_usage_stats(api_key)
            await self.anomaly_detector.check_budget_alerts(api_key, usage_stats)

    async def get_usage_stats(self, api_key: str) -> Dict[str, Any]:
        """Get usage statistics for an API key."""
        return await self.budget_manager.get_usage_stats(api_key)

    def _calculate_risk_score(
        self, intent: RequestIntent, estimated_tokens: int, max_tokens: Optional[int]
    ) -> float:
        """Calculate risk score for the request."""
        score = 0.0

        # High token usage increases risk
        if estimated_tokens > 5000:
            score += 0.3
        elif estimated_tokens > 2000:
            score += 0.1

        # Long generation requests are riskier
        if intent == RequestIntent.LONG_GENERATION:
            score += 0.4

        # Very high max_tokens increases risk
        if max_tokens and max_tokens > 4000:
            score += 0.2

        return min(1.0, score)

    def _generate_recommendations(
        self, intent: RequestIntent, risk_score: float, estimated_tokens: int
    ) -> List[str]:
        """Generate routing/handling recommendations."""
        recommendations = []

        if risk_score > 0.5:
            recommendations.append(
                "Consider routing to cheaper model due to high token usage"
            )

        if intent == RequestIntent.CODE_GENERATION:
            recommendations.append(
                "Route to code-optimized model (e.g., DeepSeek, Grok)"
            )

        if intent == RequestIntent.RAG_SUMMARIZATION:
            recommendations.append(
                "Consider using RAG-enabled model for better context handling"
            )

        if estimated_tokens > 3000:
            recommendations.append("Request may benefit from streaming response")

        return recommendations


# Global gateway service instance
_gateway_service = None


def get_gateway_service(redis_client=None) -> GatewayService:
    """Get or create gateway service instance."""
    global _gateway_service
    if _gateway_service is None:
        _gateway_service = GatewayService(redis_client)
    return _gateway_service
