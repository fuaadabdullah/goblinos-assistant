"""
Vast.ai Provider Implementation
Cost-sensitive batch jobs and spot instances
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


class VastAIProvider(BaseProvider):
    """
    Vast.ai Provider for cost-optimized workloads.

    Handles:
    - Cheap H100/A100 spot runs
    - Large hyperparameter sweeps
    - Pretraining-style batch jobs

    Security: Treats hosts as untrusted - use weight encryption.
    """

    provider_type = ProviderType.VASTAI
    capabilities = ProviderCapabilities(
        supports_inference=True,
        supports_training=True,
        supports_multi_gpu=True,
        max_gpus=8,
        available_gpu_types=[
            GPUType.RTX_3090,
            GPUType.RTX_4090,
            GPUType.A100_40GB,
            GPUType.A100_80GB,
            GPUType.H100,
        ],
        supports_spot=True,
        supports_streaming=False,
    )

    BASE_URL = "https://console.vast.ai/api/v0"

    def __init__(self):
        config = get_cloud_config()
        self.api_key = config.vastai.api_key
        self.min_rating = config.vastai.min_host_reliability
        self.min_upload_speed = config.vastai.min_internet_speed
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with auth."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
            )
        return self._client

    async def health_check(self) -> bool:
        """Check Vast.ai API health."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.BASE_URL}/users/current/", params={"api_key": self.api_key}
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning("Vast.ai health check failed", error=str(e))
            return False

    async def get_availability(self) -> float:
        """Get availability based on matching offers."""
        try:
            offers = await self._search_offers(
                gpu_type="RTX_4090",
                num_gpus=1,
                limit=10,
            )
            # Score based on number of available offers
            return min(len(offers) * 10, 100)
        except Exception:
            return 30.0  # Assume some availability

    async def get_cost_estimate(
        self,
        job_type: JobType,
        gpu_type: Optional[str] = None,
    ) -> float:
        """
        Cost estimate per hour for Vast.ai.
        Significantly cheaper than RunPod for spot instances.

        Typical spot pricing:
        - RTX 4090: ~$0.20-0.35/hr
        - A100 80GB: ~$1.00-1.50/hr
        - H100: ~$2.00-3.00/hr
        """
        # Get actual offers for accurate pricing
        try:
            offers = await self._search_offers(
                gpu_type=gpu_type or "RTX_4090",
                num_gpus=1,
                limit=5,
            )
            if offers:
                # Return median price
                prices = sorted([o.get("dph_total", 1.0) for o in offers])
                return prices[len(prices) // 2]
        except Exception:
            pass

        # Fallback estimates
        gpu_costs = {
            "rtx_3090": 0.20,
            "rtx_4090": 0.30,
            "a100_40gb": 0.90,
            "a100_80gb": 1.20,
            "h100": 2.50,
        }

        if gpu_type:
            normalized = gpu_type.lower().replace(" ", "_").replace("-", "_")
            return gpu_costs.get(normalized, 0.50)

        return 0.30  # Default RTX 4090 spot

    async def get_latency_estimate(self, job_type: JobType) -> float:
        """
        Latency estimate in milliseconds.
        Vast.ai has higher latency due to instance spin-up.
        """
        if job_type == JobType.INFERENCE:
            return 5000.0  # 5s cold start
        elif job_type == JobType.BATCH_INFERENCE:
            return 10000.0  # Batch processing setup
        return float("inf")

    async def _search_offers(
        self,
        gpu_type: str = "RTX_4090",
        num_gpus: int = 1,
        min_ram: int = 16,
        min_disk: int = 50,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search for available offers matching criteria.

        Filters:
        - Host rating >= min_rating
        - Upload speed >= min_upload_speed
        - Verified hosts preferred
        """
        client = await self._get_client()

        # Build query
        query = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "num_gpus": {"gte": num_gpus},
            "gpu_ram": {"gte": min_ram * 1024},  # MB
            "disk_space": {"gte": min_disk},
            "inet_up": {"gte": self.min_upload_speed},
            "reliability2": {"gte": self.min_rating / 5.0},  # 0-1 scale
        }

        # GPU type filter
        gpu_name_map = {
            "rtx_3090": "RTX 3090",
            "rtx_4090": "RTX 4090",
            "a100_40gb": "A100",
            "a100_80gb": "A100",
            "h100": "H100",
        }
        normalized = gpu_type.lower().replace(" ", "_").replace("-", "_")
        if normalized in gpu_name_map:
            query["gpu_name"] = {"eq": gpu_name_map[normalized]}

        response = await client.get(
            f"{self.BASE_URL}/bundles",
            params={
                "api_key": self.api_key,
                "q": str(query),
                "order": "dph_total",  # Sort by price
                "limit": limit,
            },
        )

        if response.status_code != 200:
            logger.error("Vast.ai offer search failed", status=response.status_code)
            return []

        return response.json().get("offers", [])

    async def submit_inference(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> JobResult:
        """
        Submit inference job to Vast.ai.
        Creates an instance, runs inference, and destroys.

        Note: Not ideal for real-time inference due to spin-up time.
        Better suited for batch inference.
        """
        import time

        start_time = time.time()

        # For real-time inference, recommend using GCP/RunPod
        logger.warning(
            "Vast.ai inference has high latency. Consider GCP or RunPod for real-time."
        )

        try:
            # Find cheapest offer
            offers = await self._search_offers(gpu_type="RTX_4090", limit=5)
            if not offers:
                raise RuntimeError("No suitable Vast.ai offers available")

            offer = offers[0]  # Cheapest

            # Create instance
            instance = await self._create_instance(
                offer_id=offer["id"],
                image="vllm/vllm-openai:latest",
                disk_gb=30,
                onstart=f"python -m vllm.entrypoints.openai.api_server --model {model}",
            )

            instance_id = instance.get("new_contract")

            try:
                # Wait for instance to be ready
                await self._wait_for_instance(instance_id, timeout=300)

                # Get instance IP
                status = await self._get_instance_status(instance_id)
                ip = status.get("public_ipaddr")
                port = (
                    status.get("ports", {})
                    .get("8000/tcp", [{}])[0]
                    .get("HostPort", 8000)
                )

                # Run inference
                async with httpx.AsyncClient() as inference_client:
                    response = await inference_client.post(
                        f"http://{ip}:{port}/v1/completions",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                        timeout=120.0,
                    )
                    result = response.json()

                duration = time.time() - start_time
                cost = duration * (offer.get("dph_total", 0.30) / 3600)

                return JobResult(
                    success=True,
                    provider=self.provider_type,
                    job_id=str(instance_id),
                    output=result.get("choices", [{}])[0].get("text", ""),
                    cost=cost,
                    duration_seconds=duration,
                    metadata={
                        "model": model,
                        "offer_id": offer["id"],
                        "gpu": offer.get("gpu_name"),
                    },
                )

            finally:
                # Always destroy instance
                await self._destroy_instance(instance_id)

        except Exception as e:
            logger.error("Vast.ai inference failed", error=str(e))
            return JobResult(
                success=False,
                provider=self.provider_type,
                job_id="",
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
        """
        Submit training job to Vast.ai.

        Config should include:
        - gpu_type: GPU type (e.g., "RTX_4090")
        - num_gpus: Number of GPUs
        - image: Docker image
        - script: Training script to run
        - env: Environment variables
        """
        import time

        start_time = time.time()

        try:
            gpu_type = config.get("gpu_type", "RTX_4090")
            num_gpus = config.get("num_gpus", 1)

            # Find offers
            offers = await self._search_offers(
                gpu_type=gpu_type,
                num_gpus=num_gpus,
                min_disk=config.get("disk_gb", 100),
                limit=10,
            )

            if not offers:
                raise RuntimeError(f"No Vast.ai offers for {num_gpus}x {gpu_type}")

            # Select best offer (cheapest with good reliability)
            offer = offers[0]

            # Build startup script with checkpointing
            onstart_script = f"""
#!/bin/bash
set -e

# Pull weights from GCS (encrypted)
if [ -n "$GCS_WEIGHTS_URL" ]; then
    echo "Pulling weights from GCS..."
    gsutil cp "$GCS_WEIGHTS_URL" /weights/model.bin
    if [ -n "$WEIGHT_ENCRYPTION_KEY" ]; then
        python /scripts/decrypt_weights.py /weights/model.bin
    fi
fi

# Resume from checkpoint if available
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT_PATH"
    gsutil cp -r "$CHECKPOINT_PATH" /checkpoints/
fi

# Run training
{config.get("script", 'echo "No training script specified"')}

# Upload checkpoint
gsutil cp -r /checkpoints/* gs://{config.gcp.checkpoint_bucket}/$(date +%Y%m%d_%H%M%S)/
"""

            # Create instance
            instance = await self._create_instance(
                offer_id=offer["id"],
                image=config.get(
                    "image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel"
                ),
                disk_gb=config.get("disk_gb", 100),
                onstart=onstart_script,
                env={
                    "GCS_WEIGHTS_URL": config.get("weights_url", ""),
                    "CHECKPOINT_PATH": checkpoint_path or "",
                    "WEIGHT_ENCRYPTION_KEY": "",
                    **config.get("env", {}),
                },
            )

            instance_id = instance.get("new_contract")

            return JobResult(
                success=True,
                provider=self.provider_type,
                job_id=str(instance_id),
                output={
                    "instance_id": instance_id,
                    "offer_id": offer["id"],
                    "gpu": offer.get("gpu_name"),
                    "cost_per_hour": offer.get("dph_total"),
                },
                cost=0.0,  # Will track via instance runtime
                duration_seconds=time.time() - start_time,
                metadata={
                    "gpu_type": gpu_type,
                    "num_gpus": num_gpus,
                    "offer": offer,
                },
            )

        except Exception as e:
            logger.error("Vast.ai training submission failed", error=str(e))
            return JobResult(
                success=False,
                provider=self.provider_type,
                job_id="",
                output=None,
                cost=0.0,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def _create_instance(
        self,
        offer_id: int,
        image: str,
        disk_gb: int = 50,
        onstart: str = "",
        env: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Create a new Vast.ai instance."""
        client = await self._get_client()

        payload = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "onstart": onstart,
            "runtype": "args",
        }

        if env:
            payload["env"] = env

        response = await client.put(
            f"{self.BASE_URL}/asks/{offer_id}/",
            params={"api_key": self.api_key},
            json=payload,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to create instance: {response.text}")

        return response.json()

    async def _wait_for_instance(
        self,
        instance_id: int,
        timeout: float = 300.0,
        poll_interval: float = 10.0,
    ) -> None:
        """Wait for instance to be running."""
        elapsed = 0.0

        while elapsed < timeout:
            status = await self._get_instance_status(instance_id)
            actual_status = status.get("actual_status")

            if actual_status == "running":
                return
            elif actual_status in ["error", "exited"]:
                raise RuntimeError(f"Instance failed: {status.get('status_msg')}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Instance {instance_id} did not start in {timeout}s")

    async def _get_instance_status(self, instance_id: int) -> dict[str, Any]:
        """Get instance status."""
        client = await self._get_client()

        response = await client.get(
            f"{self.BASE_URL}/instances/{instance_id}/",
            params={"api_key": self.api_key},
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to get instance status: {response.text}")

        return response.json()

    async def _destroy_instance(self, instance_id: int) -> bool:
        """Destroy/terminate an instance."""
        try:
            client = await self._get_client()

            response = await client.delete(
                f"{self.BASE_URL}/instances/{instance_id}/",
                params={"api_key": self.api_key},
            )

            return response.status_code == 200
        except Exception as e:
            logger.error(
                "Failed to destroy instance", instance_id=instance_id, error=str(e)
            )
            return False

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a Vast.ai instance."""
        try:
            return await self._get_instance_status(int(job_id))
        except Exception as e:
            return {"error": str(e)}

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel/destroy a Vast.ai instance."""
        return await self._destroy_instance(int(job_id))

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
