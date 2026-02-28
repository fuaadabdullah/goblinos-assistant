"""
GCP Provider Implementation
Development environment with Ollama + llama.cpp
"""

import asyncio
from typing import Any, Optional

import httpx
import structlog

from ...config.cloud_providers import get_cloud_config
from .base import (
    BaseProvider,
    GPUType,
    JobResult,
    JobType,
    ProviderCapabilities,
    ProviderType,
)

logger = structlog.get_logger()


class GCPProvider(BaseProvider):
    """
    GCP Provider for development and fallback inference.
    
    Uses Ollama running on GCE for:
    - Fast development iteration
    - Persistent model pulls
    - Low-latency experiments
    """
    
    provider_type = ProviderType.GCP
    capabilities = ProviderCapabilities(
        supports_inference=True,
        supports_training=False,  # Use RunPod/Vast.ai for training
        supports_multi_gpu=False,
        max_gpus=1,
        available_gpu_types=[GPUType.RTX_4090],  # Dev instance
        supports_spot=False,
        supports_streaming=True,
    )
    
    def __init__(self):
        config = get_cloud_config()
        self.ollama_host = config.gcp.ollama_internal_url or "http://localhost:11434"
        self._client: Optional[httpx.AsyncClient] = None
        self._last_health_check: Optional[bool] = None
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.ollama_host,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        return self._client
    
    async def health_check(self) -> bool:
        """Check Ollama server health."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            self._last_health_check = response.status_code == 200
            return self._last_health_check
        except Exception as e:
            logger.warning("GCP health check failed", error=str(e))
            self._last_health_check = False
            return False
    
    async def get_availability(self) -> float:
        """GCP dev environment is always available if healthy."""
        if self._last_health_check is None:
            await self.health_check()
        return 100.0 if self._last_health_check else 0.0
    
    async def get_cost_estimate(
        self,
        job_type: JobType,
        gpu_type: Optional[str] = None,
    ) -> float:
        """
        Cost estimate for GCP.
        Development environment has fixed costs.
        """
        # GCP dev instance: ~$0.50/hr for n1-standard-4 + spot GPU
        if job_type in [JobType.INFERENCE, JobType.BATCH_INFERENCE]:
            return 0.50  # Per hour
        return 0.0  # Training not supported
    
    async def get_latency_estimate(self, job_type: JobType) -> float:
        """Latency estimate in milliseconds."""
        # Ollama local inference is fast
        if job_type == JobType.INFERENCE:
            return 200.0  # 200ms typical for small models
        elif job_type == JobType.BATCH_INFERENCE:
            return 500.0
        return float("inf")
    
    async def submit_inference(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs,
    ) -> JobResult:
        """
        Submit inference request to Ollama.
        """
        import time
        import uuid
        
        start_time = time.time()
        job_id = str(uuid.uuid4())
        
        try:
            client = await self._get_client()
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            }
            
            if not stream:
                response = await client.post("/api/generate", json=payload)
                response.raise_for_status()
                result = response.json()
                
                duration = time.time() - start_time
                
                return JobResult(
                    success=True,
                    provider=self.provider_type,
                    job_id=job_id,
                    output=result.get("response", ""),
                    cost=duration * (0.50 / 3600),  # Pro-rated hourly cost
                    duration_seconds=duration,
                    metadata={
                        "model": model,
                        "tokens_generated": result.get("eval_count", 0),
                        "tokens_per_second": result.get("eval_count", 0) / duration if duration > 0 else 0,
                    }
                )
            else:
                # Streaming response
                async def stream_generator():
                    async with client.stream("POST", "/api/generate", json=payload) as response:
                        async for line in response.aiter_lines():
                            if line:
                                yield line
                
                return JobResult(
                    success=True,
                    provider=self.provider_type,
                    job_id=job_id,
                    output=stream_generator(),
                    cost=0.0,  # Will be calculated after streaming
                    duration_seconds=0.0,
                    metadata={"streaming": True, "model": model}
                )
                
        except Exception as e:
            logger.error("GCP inference failed", error=str(e), job_id=job_id)
            return JobResult(
                success=False,
                provider=self.provider_type,
                job_id=job_id,
                output=None,
                cost=0.0,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )
    
    async def submit_training(
        self,
        config: dict[str, Any],
        checkpoint_path: Optional[str] = None,
    ) -> JobResult:
        """Training not supported on GCP dev environment."""
        return JobResult(
            success=False,
            provider=self.provider_type,
            job_id="",
            output=None,
            cost=0.0,
            duration_seconds=0.0,
            error="Training not supported on GCP development environment. Use RunPod or Vast.ai.",
        )
    
    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get job status (Ollama jobs are synchronous)."""
        return {
            "job_id": job_id,
            "status": "completed",  # Ollama jobs complete immediately
            "provider": self.provider_type.value,
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel is not applicable for synchronous Ollama jobs."""
        return True
    
    async def list_models(self) -> list[dict[str, Any]]:
        """List available models in Ollama."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            return []
    
    async def pull_model(self, model: str) -> bool:
        """Pull a model into Ollama."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/pull",
                json={"name": model},
                timeout=httpx.Timeout(600.0),  # 10 min for large models
            )
            return response.status_code == 200
        except Exception as e:
            logger.error("Failed to pull model", model=model, error=str(e))
            return False
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
