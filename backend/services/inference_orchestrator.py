"""
Multi-Provider Inference Orchestrator

Routes inference requests to the optimal provider based on:
- Latency requirements
- Cost constraints
- Model availability
- Provider health and load

Providers:
- GCP (Ollama/llama.cpp): Low latency, persistent models
- RunPod: Production inference, multi-GPU
- Vast.ai: Batch inference, cost-sensitive
- Local: Development, offline fallback
"""

import os
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random

logger = logging.getLogger(__name__)


class InferenceProvider(Enum):
    """Available inference providers."""

    GCP_OLLAMA = "gcp_ollama"
    GCP_LLAMACPP = "gcp_llamacpp"
    RUNPOD = "runpod"
    VASTAI = "vastai"
    LOCAL_OLLAMA = "local_ollama"
    LOCAL_LLAMACPP = "local_llamacpp"
    KAMATERA = "kamatera"


class RoutingStrategy(Enum):
    """Routing strategies for provider selection."""

    LOWEST_LATENCY = "lowest_latency"
    LOWEST_COST = "lowest_cost"
    ROUND_ROBIN = "round_robin"
    WEIGHTED_RANDOM = "weighted_random"
    FAILOVER = "failover"


@dataclass
class ProviderHealth:
    """Health status of a provider."""

    provider: InferenceProvider
    healthy: bool
    latency_ms: float
    last_check: datetime
    error_rate: float = 0.0
    requests_per_minute: int = 0
    queue_depth: int = 0


@dataclass
class ProviderCost:
    """Cost information for a provider."""

    provider: InferenceProvider
    cost_per_1k_tokens: float
    cost_per_request: float = 0.0
    minimum_charge: float = 0.0


@dataclass
class InferenceRequest:
    """Structured inference request."""

    messages: List[Dict[str, str]]
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False
    timeout_ms: Optional[int] = None
    max_cost: Optional[float] = None
    prefer_local: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result of an inference request."""

    content: str
    provider: InferenceProvider
    model: str
    latency_ms: float
    tokens_used: int
    cost: float
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiProviderOrchestrator:
    """
    Orchestrates inference across multiple cloud providers.

    Features:
    - Intelligent routing based on latency/cost/availability
    - Health monitoring and automatic failover
    - Request queuing and load balancing
    - Cost tracking and budgeting
    - Caching for repeated queries
    """

    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.LOWEST_LATENCY,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 300,
        health_check_interval: int = 60,
    ):
        self.default_strategy = default_strategy
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl_seconds
        self.health_check_interval = health_check_interval

        # Provider health tracking
        self._health: Dict[InferenceProvider, ProviderHealth] = {}

        # Cost tracking
        self._costs: Dict[InferenceProvider, ProviderCost] = self._init_costs()

        # Request metrics
        self._request_counts: Dict[InferenceProvider, int] = {}
        self._error_counts: Dict[InferenceProvider, int] = {}

        # Cache
        self._cache: Dict[str, tuple] = {}  # (result, timestamp)

        # Provider clients (lazy loaded)
        self._clients: Dict[InferenceProvider, Any] = {}

        # Load configuration
        self._load_config()

    def _init_costs(self) -> Dict[InferenceProvider, ProviderCost]:
        """Initialize provider cost information."""
        return {
            InferenceProvider.GCP_OLLAMA: ProviderCost(
                provider=InferenceProvider.GCP_OLLAMA,
                cost_per_1k_tokens=0.0,  # Self-hosted
                cost_per_request=0.0001,  # Compute cost
            ),
            InferenceProvider.GCP_LLAMACPP: ProviderCost(
                provider=InferenceProvider.GCP_LLAMACPP,
                cost_per_1k_tokens=0.0,
                cost_per_request=0.0001,
            ),
            InferenceProvider.RUNPOD: ProviderCost(
                provider=InferenceProvider.RUNPOD,
                cost_per_1k_tokens=0.00015,  # Serverless pricing
                cost_per_request=0.001,
            ),
            InferenceProvider.VASTAI: ProviderCost(
                provider=InferenceProvider.VASTAI,
                cost_per_1k_tokens=0.0001,  # Spot pricing
                cost_per_request=0.0005,
            ),
            InferenceProvider.LOCAL_OLLAMA: ProviderCost(
                provider=InferenceProvider.LOCAL_OLLAMA,
                cost_per_1k_tokens=0.0,
                cost_per_request=0.0,
            ),
            InferenceProvider.LOCAL_LLAMACPP: ProviderCost(
                provider=InferenceProvider.LOCAL_LLAMACPP,
                cost_per_1k_tokens=0.0,
                cost_per_request=0.0,
            ),
            InferenceProvider.KAMATERA: ProviderCost(
                provider=InferenceProvider.KAMATERA,
                cost_per_1k_tokens=0.0,
                cost_per_request=0.00005,
            ),
        }

    def _load_config(self):
        """Load provider configuration from environment."""
        self._config = {
            "gcp_ollama_url": os.getenv("GCP_OLLAMA_URL", ""),
            "gcp_llamacpp_url": os.getenv("GCP_LLAMACPP_URL", ""),
            "runpod_endpoint_id": os.getenv("RUNPOD_ENDPOINT_ID", ""),
            "runpod_api_key": os.getenv("RUNPOD_API_KEY", ""),
            "vastai_api_key": os.getenv("VASTAI_API_KEY", ""),
            "local_ollama_url": os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434"),
            "local_llamacpp_url": os.getenv("LOCAL_LLAMACPP_URL", "http://localhost:8080"),
            "kamatera_url": os.getenv("KAMATERA_LLM_URL", "http://45.61.60.3:8002"),
        }

    def _get_available_providers(self) -> List[InferenceProvider]:
        """Get list of configured and healthy providers."""
        available = []

        if self._config["local_ollama_url"]:
            available.append(InferenceProvider.LOCAL_OLLAMA)

        if self._config["local_llamacpp_url"]:
            available.append(InferenceProvider.LOCAL_LLAMACPP)

        if self._config["kamatera_url"]:
            available.append(InferenceProvider.KAMATERA)

        if self._config["gcp_ollama_url"]:
            available.append(InferenceProvider.GCP_OLLAMA)

        if self._config["gcp_llamacpp_url"]:
            available.append(InferenceProvider.GCP_LLAMACPP)

        if self._config["runpod_api_key"] and self._config["runpod_endpoint_id"]:
            available.append(InferenceProvider.RUNPOD)

        if self._config["vastai_api_key"]:
            available.append(InferenceProvider.VASTAI)

        # Filter by health
        return [p for p in available if p not in self._health or self._health[p].healthy]

    async def infer(
        self,
        request: InferenceRequest,
        strategy: Optional[RoutingStrategy] = None,
    ) -> InferenceResult:
        """
        Execute inference request with intelligent routing.

        Args:
            request: The inference request
            strategy: Override routing strategy

        Returns:
            Inference result with provider info
        """
        strategy = strategy or self.default_strategy

        # Check cache first
        if self.enable_caching and not request.stream:
            cached = self._check_cache(request)
            if cached:
                return cached

        # Select provider
        provider = await self._select_provider(request, strategy)

        if not provider:
            raise RuntimeError("No available providers for inference")

        # Execute inference
        start_time = time.time()
        try:
            result = await self._execute_inference(provider, request)

            # Update metrics
            self._record_success(provider)

            # Cache result
            if self.enable_caching and not request.stream:
                self._cache_result(request, result)

            return result

        except Exception as e:
            logger.error(f"Inference failed on {provider.value}: {e}")
            self._record_error(provider)

            # Try failover
            if strategy != RoutingStrategy.FAILOVER:
                return await self.infer(request, RoutingStrategy.FAILOVER)

            raise

    async def _select_provider(
        self,
        request: InferenceRequest,
        strategy: RoutingStrategy,
    ) -> Optional[InferenceProvider]:
        """Select the best provider based on strategy."""
        available = self._get_available_providers()

        if not available:
            return None

        # Apply request preferences
        if request.prefer_local:
            local_providers = [
                p
                for p in available
                if p in (InferenceProvider.LOCAL_OLLAMA, InferenceProvider.LOCAL_LLAMACPP)
            ]
            if local_providers:
                available = local_providers

        # Apply cost constraint
        if request.max_cost is not None:
            estimated_tokens = request.max_tokens * 1.5  # Estimate total tokens
            available = [
                p for p in available if self._estimate_cost(p, estimated_tokens) <= request.max_cost
            ]

        if not available:
            return None

        # Route based on strategy
        if strategy == RoutingStrategy.LOWEST_LATENCY:
            return self._select_lowest_latency(available)

        elif strategy == RoutingStrategy.LOWEST_COST:
            return self._select_lowest_cost(available)

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available)

        elif strategy == RoutingStrategy.WEIGHTED_RANDOM:
            return self._select_weighted_random(available)

        elif strategy == RoutingStrategy.FAILOVER:
            return self._select_failover(available)

        return available[0]

    def _select_lowest_latency(
        self,
        providers: List[InferenceProvider],
    ) -> InferenceProvider:
        """Select provider with lowest latency."""
        # Priority order for latency
        priority = [
            InferenceProvider.LOCAL_OLLAMA,
            InferenceProvider.LOCAL_LLAMACPP,
            InferenceProvider.KAMATERA,
            InferenceProvider.GCP_OLLAMA,
            InferenceProvider.GCP_LLAMACPP,
            InferenceProvider.RUNPOD,
            InferenceProvider.VASTAI,
        ]

        for p in priority:
            if p in providers:
                return p

        return providers[0]

    def _select_lowest_cost(
        self,
        providers: List[InferenceProvider],
    ) -> InferenceProvider:
        """Select provider with lowest cost."""
        return min(
            providers,
            key=lambda p: self._costs[p].cost_per_1k_tokens + self._costs[p].cost_per_request,
        )

    def _select_round_robin(
        self,
        providers: List[InferenceProvider],
    ) -> InferenceProvider:
        """Select provider using round-robin."""
        counts = {p: self._request_counts.get(p, 0) for p in providers}
        return min(providers, key=lambda p: counts[p])

    def _select_weighted_random(
        self,
        providers: List[InferenceProvider],
    ) -> InferenceProvider:
        """Select provider with weighted random (by inverse latency)."""
        weights = []
        for p in providers:
            if p in self._health:
                # Inverse of latency = higher weight for faster providers
                weight = 1000.0 / max(self._health[p].latency_ms, 1)
            else:
                weight = 1.0
            weights.append(weight)

        total = sum(weights)
        r = random.random() * total

        cumsum = 0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                return providers[i]

        return providers[-1]

    def _select_failover(
        self,
        providers: List[InferenceProvider],
    ) -> InferenceProvider:
        """Select first healthy provider in priority order."""
        priority = [
            InferenceProvider.LOCAL_OLLAMA,
            InferenceProvider.LOCAL_LLAMACPP,
            InferenceProvider.KAMATERA,
            InferenceProvider.GCP_OLLAMA,
            InferenceProvider.GCP_LLAMACPP,
            InferenceProvider.RUNPOD,
            InferenceProvider.VASTAI,
        ]

        for p in priority:
            if p in providers and (p not in self._health or self._health[p].healthy):
                return p

        return providers[0] if providers else None

    async def _execute_inference(
        self,
        provider: InferenceProvider,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Execute inference on the selected provider."""
        start_time = time.time()

        if provider in (
            InferenceProvider.LOCAL_OLLAMA,
            InferenceProvider.GCP_OLLAMA,
            InferenceProvider.KAMATERA,
        ):
            result = await self._call_ollama(provider, request)

        elif provider in (
            InferenceProvider.LOCAL_LLAMACPP,
            InferenceProvider.GCP_LLAMACPP,
        ):
            result = await self._call_llamacpp(provider, request)

        elif provider == InferenceProvider.RUNPOD:
            result = await self._call_runpod(request)

        elif provider == InferenceProvider.VASTAI:
            result = await self._call_vastai(request)

        else:
            raise ValueError(f"Unknown provider: {provider}")

        latency_ms = (time.time() - start_time) * 1000

        return InferenceResult(
            content=result.get("content", ""),
            provider=provider,
            model=request.model,
            latency_ms=latency_ms,
            tokens_used=result.get("tokens", 0),
            cost=self._estimate_cost(provider, result.get("tokens", 0)),
            metadata=result.get("metadata", {}),
        )

    async def _call_ollama(
        self,
        provider: InferenceProvider,
        request: InferenceRequest,
    ) -> Dict[str, Any]:
        """Call Ollama API."""
        import httpx

        if provider == InferenceProvider.LOCAL_OLLAMA:
            url = self._config["local_ollama_url"]
        elif provider == InferenceProvider.KAMATERA:
            url = self._config["kamatera_url"]
        else:
            url = self._config["gcp_ollama_url"]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{url}/api/chat",
                json={
                    "model": request.model or "qwen2.5:3b",
                    "messages": request.messages,
                    "options": {
                        "num_predict": request.max_tokens,
                        "temperature": request.temperature,
                    },
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

            return {
                "content": data.get("message", {}).get("content", ""),
                "tokens": data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
            }

    async def _call_llamacpp(
        self,
        provider: InferenceProvider,
        request: InferenceRequest,
    ) -> Dict[str, Any]:
        """Call llama.cpp API."""
        import httpx

        if provider == InferenceProvider.LOCAL_LLAMACPP:
            url = self._config["local_llamacpp_url"]
        else:
            url = self._config["gcp_llamacpp_url"]

        # Convert messages to prompt
        prompt = self._messages_to_prompt(request.messages)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": request.max_tokens,
                    "temperature": request.temperature,
                    "stop": ["</s>", "[/INST]"],
                },
            )
            response.raise_for_status()
            data = response.json()

            return {
                "content": data.get("content", ""),
                "tokens": data.get("tokens_predicted", 0) + data.get("tokens_evaluated", 0),
            }

    async def _call_runpod(self, request: InferenceRequest) -> Dict[str, Any]:
        """Call RunPod serverless endpoint."""
        from providers.runpod_adapter import get_runpod_adapter

        adapter = get_runpod_adapter()
        endpoint_id = self._config["runpod_endpoint_id"]

        result = await adapter.chat_completion(
            endpoint_id=endpoint_id,
            messages=request.messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return {
            "content": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "tokens": result.get("usage", {}).get("total_tokens", 0),
        }

    async def _call_vastai(self, request: InferenceRequest) -> Dict[str, Any]:
        """Call Vast.ai inference endpoint."""
        # Vast.ai is designed for batch jobs and spot instance training,
        # not real-time inference. This method should never be called during
        # normal request routing.
        raise ValueError(
            "Vast.ai provider does not support real-time inference. "
            "It is reserved for batch training and task execution. "
            "Please ensure your routing strategy does not select Vast.ai for inference requests."
        )

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                prompt_parts.append(f"{content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(f" {content} </s><s>[INST] ")

        return "".join(prompt_parts)

    def _estimate_cost(self, provider: InferenceProvider, tokens: float) -> float:
        """Estimate cost for a request."""
        cost_info = self._costs.get(provider)
        if not cost_info:
            return 0.0

        return (tokens / 1000) * cost_info.cost_per_1k_tokens + cost_info.cost_per_request

    def _check_cache(self, request: InferenceRequest) -> Optional[InferenceResult]:
        """Check cache for existing result."""
        cache_key = self._get_cache_key(request)

        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.cache_ttl):
                result.cached = True
                return result
            else:
                del self._cache[cache_key]

        return None

    def _cache_result(self, request: InferenceRequest, result: InferenceResult):
        """Cache inference result."""
        cache_key = self._get_cache_key(request)
        self._cache[cache_key] = (result, datetime.utcnow())

    def _get_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request."""
        import hashlib
        import json

        key_data = json.dumps(
            {
                "messages": request.messages,
                "model": request.model,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
            sort_keys=True,
        )

        return hashlib.sha256(key_data.encode()).hexdigest()

    def _record_success(self, provider: InferenceProvider):
        """Record successful request."""
        self._request_counts[provider] = self._request_counts.get(provider, 0) + 1

    def _record_error(self, provider: InferenceProvider):
        """Record failed request."""
        self._error_counts[provider] = self._error_counts.get(provider, 0) + 1

        # Update health
        total = self._request_counts.get(provider, 0) + self._error_counts.get(provider, 0)
        if total > 0:
            error_rate = self._error_counts.get(provider, 0) / total
            if provider in self._health:
                self._health[provider].error_rate = error_rate
                if error_rate > 0.5:
                    self._health[provider].healthy = False

    async def health_check_all(self) -> Dict[InferenceProvider, ProviderHealth]:
        """Run health checks on all providers."""
        tasks = []

        for provider in self._get_available_providers():
            tasks.append(self._health_check_provider(provider))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for provider, result in zip(self._get_available_providers(), results):
            if isinstance(result, Exception):
                self._health[provider] = ProviderHealth(
                    provider=provider,
                    healthy=False,
                    latency_ms=float("inf"),
                    last_check=datetime.utcnow(),
                    error_rate=1.0,
                )
            else:
                self._health[provider] = result

        return self._health

    async def _health_check_provider(
        self,
        provider: InferenceProvider,
    ) -> ProviderHealth:
        """Health check a single provider."""
        start_time = time.time()
        healthy = False

        try:
            # Simple ping request
            request = InferenceRequest(
                messages=[{"role": "user", "content": "ping"}],
                model="",
                max_tokens=10,
            )

            await self._execute_inference(provider, request)
            healthy = True

        except Exception as e:
            logger.warning(f"Health check failed for {provider.value}: {e}")

        latency_ms = (time.time() - start_time) * 1000

        return ProviderHealth(
            provider=provider,
            healthy=healthy,
            latency_ms=latency_ms,
            last_check=datetime.utcnow(),
            error_rate=self._error_counts.get(provider, 0)
            / max(self._request_counts.get(provider, 1), 1),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "providers": {
                p.value: {
                    "requests": self._request_counts.get(p, 0),
                    "errors": self._error_counts.get(p, 0),
                    "health": self._health.get(
                        p,
                        ProviderHealth(
                            provider=p,
                            healthy=True,
                            latency_ms=0,
                            last_check=datetime.utcnow(),
                        ),
                    ).__dict__
                    if p in self._health
                    else None,
                    "cost": self._costs.get(p).__dict__ if p in self._costs else None,
                }
                for p in InferenceProvider
            },
            "cache_size": len(self._cache),
            "default_strategy": self.default_strategy.value,
        }


# Singleton instance
_orchestrator: Optional[MultiProviderOrchestrator] = None


def get_orchestrator() -> MultiProviderOrchestrator:
    """Get or create the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiProviderOrchestrator()
    return _orchestrator
