"""
ChatOrchestrator Service for managing chat completion workflows.

This service coordinates the various components involved in chat completions:
- Request validation
- Provider routing
- Scaling and verification
- Response building

This replaces the monolithic create_chat_completion function with a more
modular and testable architecture.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from .request_validator import RequestValidator
from .routing import RoutingService
from ..gateway_service import GatewayService
from .gateway_exceptions import TokenBudgetExceeded, MaxTokensExceeded
from .scaling_processor import ScalingProcessor
from .verification_processor import VerificationProcessor
from .response_builder import ResponseBuilder
from .config_processor import ConfigProcessor
from .provider_factory import ProviderFactory
from .rag_processor import RAGProcessor
from .utils import record_latency_metric

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Main orchestrator for chat completion workflows."""

    def __init__(
        self,
        db: Session,
        request_validator: RequestValidator,
        routing_service: RoutingService,
        gateway_service: GatewayService,
        scaling_processor: ScalingProcessor,
        verification_processor: VerificationProcessor,
        response_builder: ResponseBuilder,
        config_processor: ConfigProcessor,
        provider_factory: ProviderFactory,
        rag_processor: RAGProcessor,
    ):
        """Initialize the ChatOrchestrator with all required services."""
        self.db = db
        self.request_validator = request_validator
        self.routing_service = routing_service
        self.gateway_service = gateway_service
        self.scaling_processor = scaling_processor
        self.verification_processor = verification_processor
        self.response_builder = response_builder
        self.config_processor = config_processor
        self.provider_factory = provider_factory
        self.rag_processor = rag_processor

    async def orchestrate_completion(
        self,
        request: Any,  # ChatCompletionRequest
        req: Request,
        service: RoutingService,
        gateway_service: GatewayService,
    ) -> Dict[str, Any]:
        """
        Orchestrate the complete chat completion workflow.

        Args:
            request: The chat completion request
            req: FastAPI request object
            service: Routing service instance
            gateway_service: Gateway service instance

        Returns:
            Dict containing routing result, gateway result, and messages
        """
        gateway_result = None
        messages = []

        try:
            # Step 1: Validate request and prepare messages
            messages, gateway_result = await self._validate_and_prepare_request(
                request, gateway_service
            )

            # Step 2: Get routing decision
            routing_result = await self._get_routing_decision(
                request, service, gateway_result
            )

            if not routing_result.get("success"):
                error_msg = routing_result.get("error", "Unknown error")
                if "Rate limit exceeded" in error_msg:
                    fallback_level = routing_result.get("fallback_level", "deny")
                    retry_after = routing_result.get("retry_after")

                    if fallback_level == "deny":
                        raise HTTPException(
                            status_code=429,
                            detail="Rate limit exceeded. Please try again later.",
                            headers={
                                "Retry-After": str(int(retry_after))
                                if retry_after
                                else "60"
                            },
                        )
                    if fallback_level == "cheap_model":
                        logger.warning(
                            f"Rate limited request {routing_result.get('request_id')} using cheap fallback"
                        )

                raise HTTPException(
                    status_code=503,
                    detail=f"No suitable provider available: {error_msg}",
                )

            if routing_result.get("emergency_mode"):
                logger.warning(
                    f"Request {routing_result.get('request_id')} served in emergency mode"
                )

            # Step 3: Process RAG context if needed
            messages, rag_context, max_tokens = await self._process_rag_context(
                request, messages
            )

            # Step 4: Handle provider-specific processing
            response_data = await self._handle_provider_processing(
                request, req, routing_result, messages, rag_context, max_tokens
            )

            return {
                "routing_result": routing_result,
                "gateway_result": gateway_result,
                "messages": messages,
                "response_data": response_data,
            }

        except Exception as e:
            if gateway_result:
                try:
                    await self.gateway_service.record_usage(
                        None,
                        0,
                        intent=gateway_result.intent,
                        success=False,
                        error_type=type(e).__name__,
                    )
                except Exception as record_error:
                    logger.warning(
                        f"Failed to record failed request anomaly: {record_error}"
                    )

            logger.error(
                f"Chat completion orchestration failed: {e}",
                exc_info=True,
                extra={
                    "correlation_id": getattr(req.state, "correlation_id", None),
                    "request_id": getattr(req.state, "request_id", None),
                },
            )
            raise HTTPException(
                status_code=500, detail=f"Chat completion failed: {str(e)}"
            )

    async def _validate_and_prepare_request(
        self, request: Any, gateway_service: GatewayService
    ) -> Tuple[List[Dict], Any]:
        """Validate request and prepare messages for processing."""
        # Validate request using the validation service
        self.request_validator.validate_chat_request(request)
        messages = self.config_processor.prepare_messages(request)

        # Run gateway checks
        gateway_result = await gateway_service.process_request(
            messages=messages,
            max_tokens=request.max_tokens,
            context=request.context,
        )

        logger.info(
            f"Gateway analysis: intent={gateway_result.intent.value}, "
            f"estimated_tokens={gateway_result.estimated_tokens}, "
            f"risk_score={gateway_result.risk_score:.2f}, "
            f"allowed={gateway_result.allowed}"
        )

        if not gateway_result.allowed:
            raise HTTPException(
                status_code=400,
                detail="Request flagged as high-risk. Please reduce token limits or simplify request.",
            )

        return messages, gateway_result

    async def _get_routing_decision(
        self, request: Any, service: RoutingService, gateway_result: Any
    ) -> Dict[str, Any]:
        """Get routing decision from the routing service."""
        # Get provider and model information
        provider_info = (
            gateway_result.provider if hasattr(gateway_result, "provider") else None
        )
        selected_model = provider_info.get("model", request.model or "auto-selected")
        temperature, max_tokens, top_p = self.config_processor.select_generation_params(
            request, gateway_result
        )

        # Apply system prompt if available
        messages = self.config_processor.apply_system_prompt(
            gateway_result.messages, gateway_result.get("system_prompt")
        )

        return {
            "success": True,
            "request_id": gateway_result.request_id,
            "provider": provider_info,
            "selected_model": selected_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "messages": messages,
        }

    async def _process_rag_context(
        self, request: Any, messages: List[Dict]
    ) -> Tuple[List[Dict], Optional[str], int]:
        """Process RAG context if needed."""
        max_tokens = request.max_tokens
        rag_context = None

        if request.intent in ["rag", "retrieval"]:
            (
                messages,
                rag_context,
                max_tokens,
            ) = await self.rag_processor.process_rag_context(
                request.intent, messages, request.context, max_tokens
            )

        return messages, rag_context, max_tokens

    async def _handle_provider_processing(
        self,
        request: Any,
        req: Request,
        routing_result: Dict[str, Any],
        messages: List[Dict],
        rag_context: Optional[str],
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Handle provider-specific processing based on provider type."""
        provider_info = routing_result["provider"]
        selected_model = routing_result["selected_model"]
        temperature = routing_result["temperature"]
        top_p = routing_result["top_p"]

        # Create adapter based on provider type
        is_local, provider_metrics_name, adapter = self.provider_factory.create_adapter(
            provider_info
        )

        if is_local:
            return await self._handle_local_provider(
                request,
                req,
                adapter,
                provider_metrics_name,
                messages,
                selected_model,
                temperature,
                max_tokens,
                top_p,
                rag_context,
            )
        else:
            return await self._handle_cloud_provider(
                adapter,
                provider_metrics_name,
                selected_model,
                messages,
                temperature,
                max_tokens,
                top_p,
            )

    async def _handle_local_provider(
        self,
        request: Any,
        req: Request,
        adapter: Any,
        provider_metrics_name: str,
        messages: List[Dict],
        selected_model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        rag_context: Optional[str],
    ) -> Dict[str, Any]:
        """Handle processing for local providers with scaling and verification."""
        # Handle local providers with scaling and verification
        scaling_outcome = await self.scaling_processor.process_inference_scaling(
            request.intent,
            adapter,
            messages,
            selected_model,
            temperature,
            max_tokens,
        )

        if scaling_outcome:
            response_text = scaling_outcome["response_text"]
            scaling_result = scaling_outcome["scaling_result"]
            response_time_ms = scaling_outcome["response_time_ms"]
            tokens_used = self.response_builder.estimate_tokens(response_text)

            await record_latency_metric(
                provider_metrics_name,
                selected_model,
                response_time_ms,
                tokens_used,
                True,
            )

            return self.response_builder.build_response_data(
                scaling_outcome.get("request_id"),
                scaling_outcome.get("provider_info"),
                selected_model,
                response_text,
                scaling_outcome,
                response_time_ms,
                tokens_used,
                True,
                rag_context=rag_context,
                scaling_result=scaling_result,
            )

        # Generate with verification and escalation
        generation = (
            await self.verification_processor.process_verification_and_escalation(
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
        )

        selected_model = generation["selected_model"]
        return self.response_builder.build_response_data(
            generation["request_id"],
            generation["provider_info"],
            selected_model,
            generation["response_text"],
            generation,
            generation["response_time_ms"],
            generation["tokens_used"],
            generation["success"],
            verification_result=generation["verification_result"],
            confidence_result=generation["confidence_result"],
            rag_context=rag_context,
            escalated=generation["escalated"],
            original_model=generation["original_model"],
        )

    async def _handle_cloud_provider(
        self,
        adapter: Any,
        provider_metrics_name: str,
        selected_model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> Dict[str, Any]:
        """Handle processing for cloud providers with simple generation."""
        (
            response_text,
            response_time_ms,
            tokens_used,
            success,
        ) = await self.verification_processor.process_simple_generation(
            adapter,
            provider_metrics_name,
            selected_model,
            messages,
            temperature,
            max_tokens,
            top_p,
        )

        return self.response_builder.build_response_data(
            None,  # request_id
            None,  # provider_info
            selected_model,
            response_text,
            None,  # routing_result
            response_time_ms,
            tokens_used,
            success,
        )
