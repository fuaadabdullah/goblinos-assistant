from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Request

from ..errors import raise_service_unavailable
from . import config_processor
from .provider_chunk import ProviderChunk
from .provider_factory import create_adapter


@dataclass
class RoutedProvider:
    """Selected provider with normalized sync/stream execution methods."""

    name: str
    model: str
    provider_info: Dict[str, Any]
    adapter: Any
    request_id: Optional[str] = None

    async def chat_completion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        model = request_data.get("model") or self.model
        messages = request_data.get("messages") or []
        kwargs: Dict[str, Any] = {}
        if request_data.get("temperature") is not None:
            kwargs["temperature"] = request_data["temperature"]
        if request_data.get("max_tokens") is not None:
            kwargs["max_tokens"] = request_data["max_tokens"]
        if request_data.get("top_p") is not None:
            kwargs["top_p"] = request_data["top_p"]

        raw = await self.adapter.generate(messages, model=model, **kwargs)
        if not isinstance(raw, dict):
            return {
                "content": str(raw),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": model,
                "finish_reason": "stop",
                "provider": self.name,
            }

        usage_raw = raw.get("usage") or {}
        prompt_tokens = int(usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)) or 0)
        completion_tokens = int(
            usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)) or 0
        )
        total_tokens = int(usage_raw.get("total_tokens", prompt_tokens + completion_tokens) or 0)

        return {
            "content": str(raw.get("content", "") or ""),
            "usage": {
                "prompt_tokens": max(0, prompt_tokens),
                "completion_tokens": max(0, completion_tokens),
                "total_tokens": max(0, total_tokens),
            },
            "model": raw.get("model") or model,
            "finish_reason": raw.get("finish_reason") or "stop",
            "provider": self.name,
            "cost": raw.get("cost"),
        }

    async def stream_chat_completion(
        self,
        request_data: Dict[str, Any],
    ) -> AsyncGenerator[ProviderChunk, None]:
        stream_fn = getattr(self.adapter, "stream_chat_completion", None)
        if callable(stream_fn):
            stream_value = stream_fn(request_data)
            if hasattr(stream_value, "__aiter__"):
                async for chunk in stream_value:
                    yield ProviderChunk.from_mapping(chunk)
                return

            if hasattr(stream_value, "__await__"):
                resolved = await stream_value
                if hasattr(resolved, "__aiter__"):
                    async for chunk in resolved:
                        yield ProviderChunk.from_mapping(chunk)
                    return
                if isinstance(resolved, list):
                    for chunk in resolved:
                        yield ProviderChunk.from_mapping(chunk)
                    return

        completion = await self.chat_completion(request_data)
        yield ProviderChunk.from_mapping(
            {
                "content": completion.get("content", ""),
                "finish_reason": completion.get("finish_reason") or "stop",
                "usage": completion.get("usage"),
                "cost": completion.get("cost"),
            }
        )


class RoutingProviderSelector:
    """Select providers via the existing chat routing service."""

    def __init__(self, routing_service: Any):
        self._routing_service = routing_service

    async def select_for_request(
        self,
        request: Any,
        http_request: Request,
        messages: list[dict[str, str]],
        gateway_result: Any,
    ) -> tuple[RoutedProvider, Dict[str, Any]]:
        requirements = config_processor.build_requirements(request, messages, gateway_result)
        client_ip, request_path, user_id = config_processor.get_client_context(http_request)

        routing_result = await self._routing_service.route_request(
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
            error_msg = routing_result.get("error", "Unknown routing error")
            raise_service_unavailable(f"No suitable provider available: {error_msg}")

        provider_info = routing_result.get("provider") or {}
        if not isinstance(provider_info, dict):
            provider_info = {}

        provider_name = str(
            provider_info.get("name")
            or provider_info.get("provider_name")
            or provider_info.get("provider")
            or ""
        ).strip()
        if not provider_name:
            raise_service_unavailable("No suitable provider available: missing provider name")

        selected_model = str(provider_info.get("model") or request.model or "auto-selected")
        normalized_provider_info = dict(provider_info)
        normalized_provider_info["name"] = provider_name
        normalized_provider_info.setdefault("model", selected_model)

        _is_local, _provider_metrics_name, adapter = create_adapter(normalized_provider_info)

        return (
            RoutedProvider(
                name=provider_name,
                model=selected_model,
                provider_info=normalized_provider_info,
                adapter=adapter,
                request_id=routing_result.get("request_id"),
            ),
            routing_result,
        )


__all__ = ["RoutedProvider", "RoutingProviderSelector"]
