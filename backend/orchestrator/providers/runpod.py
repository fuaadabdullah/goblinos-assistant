"""
RunPod Provider Implementation
Production inference and multi-GPU training
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


class RunPodProvider(BaseProvider):
    """
    RunPod Provider for production workloads.

    Handles:
    - Serverless inference endpoints
    - Instant Clusters for distributed training
    - Real-time inference with managed networking
    """

    provider_type = ProviderType.RUNPOD
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
        supports_streaming=True,
    )

    BASE_URL = "https://api.runpod.io/v2"
    GRAPHQL_URL = "https://api.runpod.io/graphql"

    def __init__(self):
        config = get_cloud_config()
        self.api_key = config.runpod.api_key
        self.endpoint_id = ""
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with auth."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    async def health_check(self) -> bool:
        """Check RunPod API health."""
        try:
            client = await self._get_client()
            # Test with a simple GraphQL query
            response = await client.post(
                self.GRAPHQL_URL, json={"query": "query { myself { id email } }"}
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning("RunPod health check failed", error=str(e))
            return False

    async def get_availability(self) -> float:
        """Get RunPod availability based on endpoint status."""
        if not self.endpoint_id:
            return 50.0  # Serverless always has some capacity

        try:
            client = await self._get_client()
            response = await client.get(f"{self.BASE_URL}/{self.endpoint_id}/health")
            if response.status_code == 200:
                data = response.json()
                workers = data.get("workers", {})
                ready = workers.get("ready", 0)
                total = ready + workers.get("running", 0) + workers.get("pending", 0)
                return (ready / max(total, 1)) * 100
        except Exception as e:
            logger.warning("Failed to get RunPod availability", error=str(e))

        return 75.0  # Assume moderate availability

    async def get_cost_estimate(
        self,
        job_type: JobType,
        gpu_type: Optional[str] = None,
    ) -> float:
        """
        Cost estimate per hour for RunPod.

        Serverless pricing:
        - RTX 4090: ~$0.44/hr
        - A100 80GB: ~$1.99/hr
        - H100: ~$3.99/hr
        """
        gpu_costs = {
            "rtx_3090": 0.31,
            "rtx_4090": 0.44,
            "a100_40gb": 1.29,
            "a100_80gb": 1.99,
            "h100": 3.99,
        }

        if gpu_type:
            normalized = gpu_type.lower().replace(" ", "_").replace("-", "_")
            return gpu_costs.get(normalized, 1.00)

        # Default costs by job type
        if job_type == JobType.INFERENCE:
            return 0.44  # RTX 4090
        elif job_type == JobType.TRAINING:
            return 1.99  # A100 80GB
        elif job_type == JobType.FINE_TUNING:
            return 0.44  # RTX 4090 for LoRA
        return 1.00

    async def get_latency_estimate(self, job_type: JobType) -> float:
        """Latency estimate in milliseconds."""
        if job_type == JobType.INFERENCE:
            return 150.0  # Serverless cold start + inference
        elif job_type == JobType.BATCH_INFERENCE:
            return 300.0
        return float("inf")

    async def submit_inference(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> JobResult:
        """Submit inference to RunPod serverless endpoint."""
        import time
        import uuid

        start_time = time.time()
        job_id = str(uuid.uuid4())

        if not self.endpoint_id:
            return JobResult(
                success=False,
                provider=self.provider_type,
                job_id=job_id,
                output=None,
                cost=0.0,
                duration_seconds=0.0,
                error="No RunPod endpoint configured",
            )

        try:
            client = await self._get_client()

            payload = {
                "input": {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs,
                }
            }

            # Submit job (async)
            response = await client.post(
                f"{self.BASE_URL}/{self.endpoint_id}/run",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            runpod_job_id = result.get("id")

            # Poll for completion
            output = await self._poll_job(runpod_job_id)

            duration = time.time() - start_time
            cost = duration * (0.44 / 3600)  # RTX 4090 rate

            return JobResult(
                success=True,
                provider=self.provider_type,
                job_id=runpod_job_id,
                output=output.get("output", {}).get("text", ""),
                cost=cost,
                duration_seconds=duration,
                metadata={
                    "model": model,
                    "runpod_status": output.get("status"),
                },
            )

        except Exception as e:
            logger.error("RunPod inference failed", error=str(e))
            return JobResult(
                success=False,
                provider=self.provider_type,
                job_id=job_id,
                output=None,
                cost=0.0,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def _poll_job(
        self,
        job_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> dict[str, Any]:
        """Poll for job completion."""
        import asyncio

        client = await self._get_client()
        elapsed = 0.0

        while elapsed < timeout:
            response = await client.get(
                f"{self.BASE_URL}/{self.endpoint_id}/status/{job_id}"
            )
            result = response.json()
            status = result.get("status")

            if status == "COMPLETED":
                return result
            elif status in ["FAILED", "CANCELLED"]:
                raise RuntimeError(f"Job {job_id} {status}: {result.get('error')}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

    async def submit_training(
        self,
        config: dict[str, Any],
        checkpoint_path: Optional[str] = None,
    ) -> JobResult:
        """
        Submit training job to RunPod.
        Creates a pod with the training configuration.
        """
        import time

        start_time = time.time()

        try:
            client = await self._get_client()

            # Create pod via GraphQL
            mutation = """
            mutation createPod($input: PodFindAndDeployOnDemandInput!) {
                podFindAndDeployOnDemand(input: $input) {
                    id
                    name
                    desiredStatus
                }
            }
            """

            gpu_type_id = config.get("gpu_type_id", "NVIDIA GeForce RTX 4090")

            variables = {
                "input": {
                    "cloudType": "SECURE",
                    "gpuCount": config.get("gpu_count", 1),
                    "volumeInGb": config.get("volume_gb", 50),
                    "containerDiskInGb": config.get("container_disk_gb", 20),
                    "gpuTypeId": gpu_type_id,
                    "name": config.get("name", "goblin-training"),
                    "imageName": config.get(
                        "image", "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"
                    ),
                    "dockerArgs": config.get("docker_args", ""),
                    "env": [
                        {"key": "CHECKPOINT_PATH", "value": checkpoint_path or ""},
                        {"key": "GCS_BUCKET", "value": config.gcp.checkpoint_bucket},
                        *[
                            {"key": k, "value": v}
                            for k, v in config.get("env", {}).items()
                        ],
                    ],
                }
            }

            response = await client.post(
                self.GRAPHQL_URL, json={"query": mutation, "variables": variables}
            )
            response.raise_for_status()
            result = response.json()

            pod = result.get("data", {}).get("podFindAndDeployOnDemand", {})

            return JobResult(
                success=True,
                provider=self.provider_type,
                job_id=pod.get("id", ""),
                output={"pod_id": pod.get("id"), "name": pod.get("name")},
                cost=0.0,  # Will track via pod runtime
                duration_seconds=time.time() - start_time,
                metadata={
                    "gpu_type": gpu_type_id,
                    "gpu_count": config.get("gpu_count", 1),
                },
            )

        except Exception as e:
            logger.error("RunPod training submission failed", error=str(e))
            return JobResult(
                success=False,
                provider=self.provider_type,
                job_id="",
                output=None,
                cost=0.0,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a RunPod job or pod."""
        try:
            client = await self._get_client()

            # Try endpoint job status first
            if self.endpoint_id:
                response = await client.get(
                    f"{self.BASE_URL}/{self.endpoint_id}/status/{job_id}"
                )
                if response.status_code == 200:
                    return response.json()

            # Try pod status via GraphQL
            query = """
            query getPod($podId: String!) {
                pod(input: { podId: $podId }) {
                    id
                    name
                    desiredStatus
                    lastStatusChange
                    runtime {
                        uptimeInSeconds
                        gpus {
                            id
                            gpuUtilPercent
                            memoryUtilPercent
                        }
                    }
                }
            }
            """

            response = await client.post(
                self.GRAPHQL_URL, json={"query": query, "variables": {"podId": job_id}}
            )
            return response.json().get("data", {}).get("pod", {})

        except Exception as e:
            logger.error("Failed to get job status", job_id=job_id, error=str(e))
            return {"error": str(e)}

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel/terminate a RunPod job or pod."""
        try:
            client = await self._get_client()

            # Try cancelling endpoint job
            if self.endpoint_id:
                response = await client.post(
                    f"{self.BASE_URL}/{self.endpoint_id}/cancel/{job_id}"
                )
                if response.status_code == 200:
                    return True

            # Try terminating pod
            mutation = """
            mutation terminatePod($podId: String!) {
                podTerminate(input: { podId: $podId })
            }
            """

            response = await client.post(
                self.GRAPHQL_URL,
                json={"query": mutation, "variables": {"podId": job_id}},
            )
            return response.status_code == 200

        except Exception as e:
            logger.error("Failed to cancel job", job_id=job_id, error=str(e))
            return False

    # ========== Network Volume Storage (S3) ==========

    async def list_network_volumes(self) -> list[dict]:
        """List all network volumes available to the account."""
        config = get_cloud_config()
        if not config.runpod.s3_key:
            logger.warning("RUNPOD_S3_KEY not configured - network volumes unavailable")
            return []

        try:
            client = await self._get_client()
            query = """
            query {
                myself {
                    networkVolumes {
                        id
                        name
                        size
                        dataCenterId
                    }
                }
            }
            """
            response = await client.post(self.GRAPHQL_URL, json={"query": query})
            if response.status_code == 200:
                data = response.json()
                return data.get("data", {}).get("myself", {}).get("networkVolumes", [])
        except Exception as e:
            logger.error("Failed to list network volumes", error=str(e))
        return []

    async def create_network_volume(
        self, name: str, size_gb: int = 50, datacenter_id: str = "US-TX-3"
    ) -> Optional[dict]:
        """Create a network volume for model/checkpoint storage."""
        config = get_cloud_config()
        if not config.runpod.s3_key:
            logger.error("RUNPOD_S3_KEY required for network volume creation")
            return None

        try:
            client = await self._get_client()
            mutation = """
            mutation createNetworkVolume($input: CreateNetworkVolumeInput!) {
                createNetworkVolume(input: $input) {
                    id
                    name
                    size
                    dataCenterId
                }
            }
            """
            variables = {
                "input": {
                    "name": name,
                    "size": size_gb,
                    "dataCenterId": datacenter_id,
                }
            }
            response = await client.post(
                self.GRAPHQL_URL, json={"query": mutation, "variables": variables}
            )
            if response.status_code == 200:
                data = response.json()
                volume = data.get("data", {}).get("createNetworkVolume")
                logger.info(
                    "Network volume created",
                    volume_id=volume.get("id") if volume else None,
                )
                return volume
        except Exception as e:
            logger.error("Failed to create network volume", error=str(e))
        return None

    async def get_storage_presigned_url(
        self, volume_id: str, path: str, operation: str = "PUT"
    ) -> Optional[str]:
        """Get presigned URL for uploading/downloading from network volume."""
        config = get_cloud_config()
        if not config.runpod.s3_key:
            logger.error("RUNPOD_S3_KEY required for presigned URLs")
            return None

        try:
            # RunPod uses S3-compatible storage
            # Format: https://<volume_id>.runpodnetwork.io/<path>
            s3_endpoint = f"https://{volume_id}.runpodnetwork.io"

            # Simple presigned URL (RunPod's S3-compatible API)
            presigned_url = f"{s3_endpoint}/{path}?X-RunPod-Key={config.runpod.s3_key}&op={operation}"
            return presigned_url

        except Exception as e:
            logger.error("Failed to generate presigned URL", error=str(e))
        return None

    async def upload_model_to_volume(
        self, volume_id: str, local_path: str, remote_path: str
    ) -> bool:
        """Upload a model file to a network volume."""
        config = get_cloud_config()
        if not config.runpod.s3_key:
            logger.error("RUNPOD_S3_KEY required for model upload")
            return False

        try:
            import aiofiles

            presigned_url = await self.get_storage_presigned_url(
                volume_id, remote_path, "PUT"
            )
            if not presigned_url:
                return False

            client = await self._get_client()
            async with aiofiles.open(local_path, "rb") as f:
                content = await f.read()

            response = await client.put(presigned_url, content=content)
            if response.status_code in (200, 201):
                logger.info(
                    "Model uploaded to volume", volume_id=volume_id, path=remote_path
                )
                return True
            else:
                logger.error("Upload failed", status=response.status_code)

        except Exception as e:
            logger.error("Failed to upload model", error=str(e))
        return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
