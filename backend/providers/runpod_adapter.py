"""
RunPod Provider Adapter

Provides integration with RunPod for:
- Serverless inference endpoints
- Instant Clusters for distributed training
- GPU pod management

RunPod is used for production inference with reproducible latency
and managed networking.
"""

import os
import asyncio
import logging
import httpx
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RunPodJobStatus(Enum):
    """Status of a RunPod job."""

    QUEUED = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class RunPodJob:
    """Represents a RunPod job."""

    job_id: str
    endpoint_id: str
    status: RunPodJobStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None


@dataclass
class RunPodPod:
    """Represents a RunPod GPU pod."""

    pod_id: str
    name: str
    gpu_type: str
    gpu_count: int
    status: str
    cost_per_hour: float
    image: str
    ports: Dict[str, int]


class RunPodAdapter:
    """
    Adapter for RunPod serverless and pod-based inference.

    Features:
    - Serverless endpoint management
    - Instant Cluster creation for distributed training
    - Cost tracking and limits
    - Automatic retries with exponential backoff
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 300.0,
    ):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY", "")
        self.api_base = "https://api.runpod.io/v2"
        self.graphql_base = "https://api.runpod.io/graphql"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        if not self.api_key:
            logger.warning("RunPod API key not configured")

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _make_request(
        self,
        method: str,
        url: str,
        json_data: Optional[Dict] = None,
        retries: int = 0,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.request(
                    method,
                    url,
                    headers=self._get_headers(),
                    json=json_data,
                )

                if response.status_code == 429:
                    # Rate limited - retry with backoff
                    if retries < self.max_retries:
                        delay = self.retry_delay * (2**retries)
                        logger.warning(f"RunPod rate limit hit, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        return await self._make_request(
                            method, url, json_data, retries + 1
                        )
                    raise Exception("RunPod rate limit exceeded")

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException:
                if retries < self.max_retries:
                    delay = self.retry_delay * (2**retries)
                    logger.warning(f"RunPod timeout, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    return await self._make_request(method, url, json_data, retries + 1)
                raise

    async def _graphql_query(
        self, query: str, variables: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute GraphQL query."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.graphql_base,
                headers=self._get_headers(),
                json={"query": query, "variables": variables or {}},
            )
            response.raise_for_status()
            result = response.json()
            if "errors" in result:
                raise Exception(f"GraphQL error: {result['errors']}")
            return result.get("data", {})

    # ==================== Serverless Endpoints ====================

    async def run_serverless(
        self,
        endpoint_id: str,
        input_data: Dict[str, Any],
        webhook_url: Optional[str] = None,
    ) -> RunPodJob:
        """
        Run a serverless inference job.

        Args:
            endpoint_id: RunPod endpoint ID
            input_data: Input payload for the model
            webhook_url: Optional webhook for async results

        Returns:
            RunPodJob with job details
        """
        url = f"{self.api_base}/{endpoint_id}/run"
        payload = {"input": input_data}
        if webhook_url:
            payload["webhook"] = webhook_url

        result = await self._make_request("POST", url, payload)

        return RunPodJob(
            job_id=result.get("id", ""),
            endpoint_id=endpoint_id,
            status=RunPodJobStatus.QUEUED,
            input_data=input_data,
            created_at=datetime.utcnow(),
        )

    async def run_sync(
        self,
        endpoint_id: str,
        input_data: Dict[str, Any],
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Run serverless inference synchronously (blocks until complete).

        Args:
            endpoint_id: RunPod endpoint ID
            input_data: Input payload for the model
            timeout: Maximum wait time in seconds

        Returns:
            Model output
        """
        url = f"{self.api_base}/{endpoint_id}/runsync"
        payload = {"input": input_data}

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers=self._get_headers(),
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "COMPLETED":
                return result.get("output", {})
            else:
                raise Exception(f"Job failed: {result.get('error', 'Unknown error')}")

    async def get_job_status(self, endpoint_id: str, job_id: str) -> RunPodJob:
        """Get status of a serverless job."""
        url = f"{self.api_base}/{endpoint_id}/status/{job_id}"
        result = await self._make_request("GET", url)

        return RunPodJob(
            job_id=job_id,
            endpoint_id=endpoint_id,
            status=RunPodJobStatus(result.get("status", "FAILED")),
            input_data={},
            output_data=result.get("output"),
            error=result.get("error"),
            execution_time_ms=result.get("executionTime"),
        )

    async def cancel_job(self, endpoint_id: str, job_id: str) -> bool:
        """Cancel a running job."""
        url = f"{self.api_base}/{endpoint_id}/cancel/{job_id}"
        result = await self._make_request("POST", url)
        return result.get("status") == "CANCELLED"

    # ==================== Pod Management ====================

    async def list_pods(self) -> List[RunPodPod]:
        """List all GPU pods."""
        query = """
        query {
            myself {
                pods {
                    id
                    name
                    desiredStatus
                    gpuCount
                    costPerHr
                    machine {
                        gpuDisplayName
                    }
                    runtime {
                        ports {
                            ip
                            privatePort
                            publicPort
                        }
                    }
                    imageName
                }
            }
        }
        """
        result = await self._graphql_query(query)
        pods = result.get("myself", {}).get("pods", [])

        return [
            RunPodPod(
                pod_id=pod["id"],
                name=pod["name"],
                gpu_type=pod.get("machine", {}).get("gpuDisplayName", ""),
                gpu_count=pod.get("gpuCount", 1),
                status=pod.get("desiredStatus", ""),
                cost_per_hour=float(pod.get("costPerHr", 0)),
                image=pod.get("imageName", ""),
                ports={
                    str(p["privatePort"]): p.get("publicPort")
                    for p in pod.get("runtime", {}).get("ports", [])
                },
            )
            for pod in pods
        ]

    async def create_pod(
        self,
        name: str,
        image: str,
        gpu_type: str = "NVIDIA GeForce RTX 4090",
        gpu_count: int = 1,
        volume_size_gb: int = 50,
        container_disk_gb: int = 20,
        ports: str = "8888/http,22/tcp",
        env_vars: Optional[Dict[str, str]] = None,
        volume_mount_path: str = "/workspace",
    ) -> RunPodPod:
        """
        Create a new GPU pod.

        Args:
            name: Pod name
            image: Docker image to run
            gpu_type: GPU type (e.g., "NVIDIA GeForce RTX 4090", "NVIDIA A100 80GB")
            gpu_count: Number of GPUs
            volume_size_gb: Persistent volume size
            container_disk_gb: Container disk size
            ports: Ports to expose (format: "port/protocol")
            env_vars: Environment variables
            volume_mount_path: Where to mount the volume

        Returns:
            Created pod details
        """
        mutation = """
        mutation createPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                desiredStatus
                gpuCount
                costPerHr
                machine {
                    gpuDisplayName
                }
                imageName
            }
        }
        """

        variables = {
            "input": {
                "name": name,
                "imageName": image,
                "gpuTypeId": gpu_type,
                "gpuCount": gpu_count,
                "volumeInGb": volume_size_gb,
                "containerDiskInGb": container_disk_gb,
                "ports": ports,
                "volumeMountPath": volume_mount_path,
            }
        }

        if env_vars:
            variables["input"]["env"] = [
                {"key": k, "value": v} for k, v in env_vars.items()
            ]

        result = await self._graphql_query(mutation, variables)
        pod_data = result.get("podFindAndDeployOnDemand", {})

        return RunPodPod(
            pod_id=pod_data.get("id", ""),
            name=pod_data.get("name", name),
            gpu_type=pod_data.get("machine", {}).get("gpuDisplayName", gpu_type),
            gpu_count=gpu_count,
            status=pod_data.get("desiredStatus", ""),
            cost_per_hour=float(pod_data.get("costPerHr", 0)),
            image=image,
            ports={},
        )

    async def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a pod."""
        mutation = """
        mutation terminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        result = await self._graphql_query(mutation, {"input": {"podId": pod_id}})
        return result.get("podTerminate") is not None

    async def stop_pod(self, pod_id: str) -> bool:
        """Stop a pod (can be resumed)."""
        mutation = """
        mutation stopPod($input: PodStopInput!) {
            podStop(input: $input)
        }
        """
        result = await self._graphql_query(mutation, {"input": {"podId": pod_id}})
        return result.get("podStop") is not None

    async def resume_pod(self, pod_id: str) -> bool:
        """Resume a stopped pod."""
        mutation = """
        mutation resumePod($input: PodResumeInput!) {
            podResume(input: $input) {
                id
            }
        }
        """
        result = await self._graphql_query(mutation, {"input": {"podId": pod_id}})
        return result.get("podResume") is not None

    # ==================== Instant Clusters ====================

    async def create_cluster(
        self,
        name: str,
        gpu_type: str,
        gpu_count_per_node: int,
        node_count: int,
        image: str,
        volume_size_gb: int = 100,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Create an Instant Cluster for distributed training.

        Args:
            name: Cluster name
            gpu_type: GPU type per node
            gpu_count_per_node: GPUs per node
            node_count: Number of nodes
            image: Docker image
            volume_size_gb: Volume size per node
            env_vars: Environment variables

        Returns:
            Cluster details
        """
        # Create multiple pods with networking enabled
        pods = []
        for i in range(node_count):
            pod = await self.create_pod(
                name=f"{name}-node-{i}",
                image=image,
                gpu_type=gpu_type,
                gpu_count=gpu_count_per_node,
                volume_size_gb=volume_size_gb,
                env_vars={
                    **(env_vars or {}),
                    "CLUSTER_NAME": name,
                    "NODE_RANK": str(i),
                    "NODE_COUNT": str(node_count),
                    "MASTER_ADDR": f"{name}-node-0",  # First node is master
                },
            )
            pods.append(pod)

        return {
            "cluster_name": name,
            "node_count": node_count,
            "total_gpus": gpu_count_per_node * node_count,
            "pods": [p.pod_id for p in pods],
            "master_pod": pods[0].pod_id if pods else None,
        }

    async def terminate_cluster(self, cluster_name: str) -> int:
        """Terminate all pods in a cluster."""
        pods = await self.list_pods()
        terminated = 0

        for pod in pods:
            if pod.name.startswith(f"{cluster_name}-node-"):
                await self.terminate_pod(pod.pod_id)
                terminated += 1

        return terminated

    # ==================== Inference Helpers ====================

    async def chat_completion(
        self,
        endpoint_id: str,
        messages: List[Dict[str, str]],
        model: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Run chat completion on a RunPod endpoint.

        Compatible with OpenAI API format.
        """
        input_data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if model:
            input_data["model"] = model

        if stream:
            # For streaming, use async job
            job = await self.run_serverless(endpoint_id, input_data)
            return {"job_id": job.job_id, "status": "streaming"}

        return await self.run_sync(endpoint_id, input_data)

    async def embeddings(
        self,
        endpoint_id: str,
        texts: List[str],
        model: str = "",
    ) -> List[List[float]]:
        """Generate embeddings using a RunPod endpoint."""
        input_data = {"texts": texts}
        if model:
            input_data["model"] = model

        result = await self.run_sync(endpoint_id, input_data)
        return result.get("embeddings", [])

    # ==================== Health & Monitoring ====================

    async def health_check(self, endpoint_id: str) -> Dict[str, Any]:
        """Check health of a serverless endpoint."""
        url = f"{self.api_base}/{endpoint_id}/health"
        try:
            result = await self._make_request("GET", url)
            return {
                "healthy": True,
                "workers": result.get("workers", {}),
                "jobs_completed": result.get("jobs", {}).get("completed", 0),
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
            }

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get account usage statistics."""
        query = """
        query {
            myself {
                id
                currentSpend
                serverlessDiscount
            }
        }
        """
        result = await self._graphql_query(query)
        myself = result.get("myself", {})

        return {
            "current_spend": float(myself.get("currentSpend", 0)),
            "serverless_discount": float(myself.get("serverlessDiscount", 0)),
        }


# Singleton instance
_adapter: Optional[RunPodAdapter] = None


def get_runpod_adapter() -> RunPodAdapter:
    """Get or create the RunPod adapter singleton."""
    global _adapter
    if _adapter is None:
        _adapter = RunPodAdapter()
    return _adapter
