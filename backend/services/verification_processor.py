"""
Verification processor service for handling output verification and escalation.
Integrates with the existing output verification service.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import time

from config import settings

logger = logging.getLogger(__name__)


async def process_verification_and_escalation(
    adapter: Any,
    provider_metrics_name: str,
    request: Any,
    messages: List[Dict[str, str]],
    selected_model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    req: Any,
) -> Dict[str, Any]:
    """Process output verification and handle escalation if needed."""
    verification_pipeline = None
    if request.enable_verification or request.enable_confidence_scoring:
        try:
            from .output_verification import VerificationPipeline
            verification_pipeline = VerificationPipeline(adapter)
        except ImportError:
            logger.warning("Output verification service not available")
            verification_pipeline = None

    original_model = selected_model
    escalated = False
    max_escalations = 2
    escalation_count = 0

    response_text = None
    verification_result = None
    confidence_result = None
    response_time_ms = None
    tokens_used = 0
    success = False

    while escalation_count <= max_escalations:
        start_time = time.time()
        error_type = None
        try:
            response_text = await adapter.chat(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            success = True
        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise
        finally:
            response_time_ms = (time.time() - start_time) * 1000
            tokens_used = estimate_tokens(response_text) if response_text else 0
            await _record_latency_metric(
                provider_metrics_name,
                selected_model,
                response_time_ms,
                tokens_used,
                success,
                error_type=error_type,
            )

        if not verification_pipeline:
            break

        (
            verification_result,
            confidence_result,
        ) = await verification_pipeline.verify_and_score(
            original_prompt=messages[-1]["content"],
            model_output=response_text,
            model_used=selected_model,
            context={
                "intent": request.intent,
                "latency_target": request.latency_target,
            },
            skip_verification=not request.enable_verification,
        )

        if verification_pipeline.should_reject_output(
            verification_result, confidence_result
        ):
	            from errors import raise_problem
	            raise_problem(
	                status=422,
	                title="Output Rejected",
	                detail="Output rejected due to safety or quality concerns",
	                type_uri="https://goblin-backend.onrender.com/errors/output-rejected",
	                code="OUTPUT_REJECTED",
	                errors={
	                    "verification": {
	                        "is_safe": verification_result.is_safe,
	                        "safety_score": verification_result.safety_score,
	                        "issues": verification_result.issues,
                    },
                    "confidence": {
                        "score": confidence_result.confidence_score,
                        "reasoning": confidence_result.reasoning,
                    },
                },
            )

        if request.auto_escalate and verification_pipeline.should_escalate(
            verification_result, confidence_result, selected_model
        ):
            next_model = verification_pipeline.get_escalation_target(selected_model)

            if next_model and escalation_count < max_escalations:
                logger.info(
                    f"Escalating from {selected_model} to {next_model} "
                    f"(confidence: {confidence_result.confidence_score:.2f})",
                    extra={
                        "correlation_id": getattr(req.state, "correlation_id", None),
                        "request_id": getattr(req.state, "request_id", None),
                    },
                )
                selected_model = next_model
                escalated = True
                escalation_count += 1
                continue

        break

    return {
        "response_text": response_text,
        "verification_result": verification_result,
        "confidence_result": confidence_result,
        "response_time_ms": response_time_ms,
        "tokens_used": tokens_used,
        "success": success,
        "escalated": escalated,
        "original_model": original_model,
        "selected_model": selected_model,
    }


async def process_simple_generation(
    adapter: Any,
    provider_metrics_name: str,
    selected_model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> Tuple[Optional[str], float, int, bool]:
    """Process simple generation without verification."""
    response_text = None
    success = False
    error_type = None
    start_time = time.time()

    try:
        response_text = await adapter.chat(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        success = True
    except Exception as e:
        success = False
        error_type = type(e).__name__
        raise
    finally:
        response_time_ms = (time.time() - start_time) * 1000
        tokens_used = estimate_tokens(response_text) if response_text else 0
        await _record_latency_metric(
            provider_metrics_name,
            selected_model,
            response_time_ms,
            tokens_used,
            success,
            error_type=error_type,
        )

    return response_text, response_time_ms, tokens_used, success


def estimate_tokens(text: Optional[str]) -> int:
    """Estimate token count for text."""
    if not text:
        return 0
    # Simple token estimation - 4 characters per token
    return len(text) // 4


async def _record_latency_metric(
    provider_name: str,
    model_name: str,
    response_time_ms: float,
    tokens_used: int,
    success: bool,
    error_type: Optional[str] = None,
) -> None:
    """Record latency metric for monitoring."""
    try:
        from .latency_monitoring_service import LatencyMonitoringService
        latency_monitor = LatencyMonitoringService()
        await latency_monitor.record_metric(
            provider_name=provider_name,
            model_name=model_name,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            success=success,
            error_type=error_type,
        )
    except Exception as e:
        logger.warning(f"Failed to record latency metric: {e}")


class VerificationProcessor:
    """
    Wrapper to expose verification helpers as an instantiable service.

    ChatOrchestrator expects an object with process_verification_and_escalation
    and process_simple_generation methods; this class delegates to the existing
    module-level coroutines to maintain compatibility.
    """

    async def process_verification_and_escalation(
        self,
        adapter: Any,
        provider_metrics_name: str,
        request: Any,
        messages: List[Dict[str, str]],
        selected_model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        req: Any,
    ) -> Dict[str, Any]:
        return await process_verification_and_escalation(
            adapter,
            provider_metrics_name,
            request,
            messages,
            selected_model,
            temperature,
            max_tokens,
            top_p,
            req,
        )

    async def process_simple_generation(
        self,
        adapter: Any,
        provider_metrics_name: str,
        selected_model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> Tuple[Optional[str], float, int, bool]:
        return await process_simple_generation(
            adapter,
            provider_metrics_name,
            selected_model,
            messages,
            temperature,
            max_tokens,
            top_p,
        )
