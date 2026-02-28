"""
Chat Controller for orchestrating chat completion requests.
Reduces cognitive complexity by separating orchestration logic from the API endpoint.
"""

from typing import Dict, Any, List, Tuple
from fastapi import Request
import logging

logger = logging.getLogger(__name__)


class ChatController:
    """
    Controller for chat completion orchestration.

    Handles the high-level flow of:
    1. Gateway validation and message preparation
    2. Routing decision
    3. Provider execution (local or cloud)
    4. Response building
    """

    def __init__(self):
        # Stateless controller - no instance state needed.
        # All dependencies are passed as method arguments for flexibility and testability.
        pass

    async def _check_gateway_and_prepare(
        self, request: Any, gateway_service: Any
    ) -> Tuple[List[Dict], Any]:
        """Validate request and run gateway checks."""
        from . import chat_controller_helpers

        return await chat_controller_helpers.gateway_check_and_prepare(request, gateway_service)

    async def orchestrate_completion(
        self,
        request: Any,  # ChatCompletionRequest
        req: Request,
        routing_service: Any,
        gateway_service: Any,
    ) -> Dict[str, Any]:
        """
        Orchestrate a complete chat completion request.

        Args:
            request: The chat completion request
            req: The FastAPI request object
            routing_service: The routing service instance
            gateway_service: The gateway service instance

        Returns:
            Dict containing the response data
        """
        # Import here to avoid circular imports
        from . import config_processor

        # Step 1: Gateway check and message preparation
        (
            messages,
            gateway_result,
        ) = await self._check_gateway_and_prepare(request, gateway_service)

        # Step 2: Build requirements and get client context
        requirements = config_processor.build_requirements(request, messages, gateway_result)
        client_ip, request_path, user_id = config_processor.get_client_context(req)

        # Step 3: Route the request
        routing_result = await routing_service.route_request(
            capability="chat",
            requirements=requirements,
            sla_target_ms=request.sla_target_ms,
            cost_budget=request.cost_budget,
            latency_priority=request.latency_target,
            client_ip=client_ip,
            user_id=user_id,
            request_path=request_path,
        )

        if not routing_result.get("success"):
            return {
                "success": False,
                "error": routing_result.get("error", "Unknown error"),
                "routing_result": routing_result,
                "gateway_result": gateway_result,
            }

        # Step 4: Execute the request with the selected provider
        execution_result = await self._execute_provider_request(
            request, req, routing_result, messages, gateway_result
        )

        return execution_result

    async def _execute_provider_request(
        self,
        request: Any,
        req: Request,
        routing_result: Dict[str, Any],
        messages: List[Dict],
        gateway_result: Any = None,
    ) -> Dict[str, Any]:
        """
        Execute the request with the selected provider (local or cloud).

        Returns:
            Dict containing the response data
        """
        from . import (
            config_processor,
            rag_processor,
            provider_factory,
        )

        # Get provider and model information
        provider_info = routing_result["provider"]
        selected_model = provider_info.get("model", request.model or "auto-selected")
        temperature, max_tokens, top_p = config_processor.select_generation_params(
            request, routing_result
        )

        # Apply system prompt if available
        processed_messages = config_processor.apply_system_prompt(
            messages, routing_result.get("system_prompt")
        )

        # Handle RAG processing
        (
            processed_messages,
            rag_context,
            max_tokens,
        ) = await rag_processor.process_rag_context(
            request.intent, processed_messages, request.context, max_tokens
        )

        # Create adapter based on provider type
        is_local, provider_metrics_name, adapter = provider_factory.create_adapter(provider_info)

        if is_local:
            return await self._execute_local_provider(
                request,
                req,
                routing_result,
                processed_messages,
                rag_context,
                provider_info,
                selected_model,
                temperature,
                max_tokens,
                top_p,
                adapter,
                provider_metrics_name,
                gateway_result,  # Added
            )
        else:
            return await self._execute_cloud_provider(
                request,
                routing_result,
                processed_messages,
                rag_context,  # Added
                provider_info,
                selected_model,
                temperature,
                max_tokens,
                top_p,
                adapter,
                provider_metrics_name,
                gateway_result,  # Added
            )

    async def _execute_local_provider(
        self,
        request: Any,
        req: Request,
        routing_result: Dict[str, Any],
        messages: List[Dict],
        rag_context: Any,
        provider_info: Dict[str, Any],
        selected_model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        adapter: Any,
        provider_metrics_name: str,
        _gateway_result: Any = None,  # Reserved for future use
    ) -> Dict[str, Any]:
        """
        Execute request with a local provider.
        """
        from . import (
            scaling_processor,
            verification_processor,
            response_builder,
            utils,
        )

        # Handle local providers with scaling and verification
        scaling_outcome = await scaling_processor.process_inference_scaling(
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
            tokens_used = response_builder.estimate_tokens(response_text)

            # Record latency metrics
            try:
                import sys

                utils_module = sys.modules.get("backend.services.utils", utils)
                await utils_module._record_latency_metric(
                    provider_metrics_name,
                    selected_model,
                    response_time_ms,
                    tokens_used,
                    True,
                )
            except AttributeError:
                pass

            response_data = response_builder.build_response_data(
                routing_result["request_id"],
                provider_info,
                selected_model,
                response_text,
                routing_result,
                response_time_ms,
                tokens_used,
                True,
                rag_context=rag_context,
                scaling_result=scaling_result,
            )
            return response_data

        # Generate with verification and escalation
        generation = await verification_processor.process_verification_and_escalation(
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

        actual_selected_model = generation["selected_model"]
        response_data = response_builder.build_response_data(
            routing_result["request_id"],
            provider_info,
            actual_selected_model,
            generation["response_text"],
            routing_result,
            generation["response_time_ms"],
            generation["tokens_used"],
            generation["success"],
            verification_result=generation["verification_result"],
            confidence_result=generation["confidence_result"],
            rag_context=rag_context,
            escalated=generation["escalated"],
            original_model=generation["original_model"],
        )
        return response_data

    async def _execute_cloud_provider(
        self,
        request: Any,
        routing_result: Dict[str, Any],
        messages: List[Dict],
        _rag_context: Any,  # Reserved for future use
        provider_info: Dict[str, Any],
        selected_model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        adapter: Any,
        provider_metrics_name: str,
        _gateway_result: Any = None,  # Reserved for future use
    ) -> Dict[str, Any]:
        """
        Execute request with a cloud provider.
        """
        from . import verification_processor, response_builder

        # Handle cloud providers with simple generation
        (
            response_text,
            response_time_ms,
            tokens_used,
            success,
        ) = await verification_processor.process_simple_generation(
            adapter,
            provider_metrics_name,
            selected_model,
            messages,
            temperature,
            max_tokens,
            top_p,
        )

        response_data = response_builder.build_response_data(
            routing_result["request_id"],
            provider_info,
            selected_model,
            response_text or "",
            routing_result,
            response_time_ms,
            tokens_used,
            success,
        )

        return response_data
