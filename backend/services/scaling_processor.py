"""
Scaling processor service for handling inference-time scaling.
Integrates with the existing inference scaling service.
"""

from typing import Dict, Any, Optional, List
import logging
import time

from config import settings

logger = logging.getLogger(__name__)


def should_use_scaling(
    request_intent: Optional[str], messages: List[Dict[str, str]]
) -> bool:
    """
    Determine if inference-time scaling should be used for this request.

    Uses scaling for complex queries that are likely to benefit from multiple
    candidate generation and PRM-based selection to reduce hallucinations.
    """
    # Always use scaling for explicit complex intents
    if request_intent in ["explain", "analyze", "solve", "reason"]:
        return True

    # Use scaling for long queries (>200 words) that might be complex
    user_query = messages[-1]["content"] if messages else ""
    word_count = len(user_query.split())

    if word_count > 200:
        return True

    # Use scaling for queries with complex keywords (but avoid false positives)
    complex_keywords = [
        "explain",
        "analyze",
        "why",
        "solve",
        "reason",
        "compare",
        "evaluate",
        "design",
        "implement",
        "optimize",
        "troubleshoot",
        "debug",
        "architecture",
        "system",
        "complex",
        "difficult",
        "algorithm",
        "theory",
        "mathematical",
        "scientific",
    ]

    query_lower = user_query.lower()

    # Check for complex keywords, but require minimum length to avoid greetings
    has_complex_keyword = any(keyword in query_lower for keyword in complex_keywords)
    if has_complex_keyword and len(user_query.split()) > 3:
        return True

    # Use scaling for multi-part questions (containing ?, and, or)
    if "?" in user_query and (" and " in query_lower or " or " in query_lower):
        return True

    return False


async def process_inference_scaling(
    request_intent: Optional[str],
    adapter: Any,
    messages: List[Dict[str, str]],
    selected_model: str,
    temperature: float,
    max_tokens: int,
) -> Optional[Dict[str, Any]]:
    """Process inference-time scaling if needed."""
    if not should_use_scaling(request_intent, messages):
        return None

    try:
        from .inference_scaling_service import InferenceScalingService
    except ImportError:
        logger.warning("Inference scaling service not available")
        return None

    logger.info("Using inference-time scaling for complex query")
    scaling_service = InferenceScalingService()
    user_query = messages[-1]["content"] if messages else ""

    start_time = time.time()
    scaling_result = await scaling_service.scale_inference(
        query=user_query,
        adapter=adapter,
        model=selected_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response_time_ms = (time.time() - start_time) * 1000

    if scaling_result.get("success"):
        return {
            "response_text": scaling_result["best_chain"]["response"],
            "scaling_result": scaling_result,
            "response_time_ms": response_time_ms,
        }

    logger.warning("Inference scaling failed, falling back to standard generation")
    return None


class ScalingProcessor:
    """
    Lightweight wrapper used by the service container.

    Provides instance methods that delegate to the module-level helpers so
    existing code paths expecting a class continue to work.
    """

    def should_use_scaling(
        self, request_intent: Optional[str], messages: List[Dict[str, str]]
    ) -> bool:
        return should_use_scaling(request_intent, messages)

    async def process(
        self,
        request_intent: Optional[str],
        adapter: Any,
        messages: List[Dict[str, str]],
        selected_model: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        return await process_inference_scaling(
            request_intent, adapter, messages, selected_model, temperature, max_tokens
        )
