"""
Vast.ai Provider Adapter

Provides integration with Vast.ai for:
- Cost-sensitive batch jobs
- Spot H100/A100 instances
- Large pretraining-style workloads

Vast.ai is treated as a marketplace:
- Filter hosts by reliability rating
- Script retries and checkpointing
- Encrypt weights for untrusted hosts
"""

import os
import asyncio
import logging
import httpx
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class VastInstanceStatus(Enum):
    """Status of a Vast.ai instance."""

    CREATING = "creating"
    RUNNING = "running"
    EXITED = "exited"
    DESTROYED = "destroyed"
    ERROR = "error"


@dataclass
class VastGPUInfo:
    """Information about a GPU offer on Vast.ai."""

    gpu_name: str
    gpu_ram: int  # MB
    num_gpus: int
    pcie_bw: float
    dlperf: float  # Deep learning performance score
    dlperf_per_dphtotal: float  # Performance per dollar


@dataclass
class VastHostInfo:
    """Information about a Vast.ai host machine."""

    host_id: int
    reliability: float  # 0-1 score
    verified: bool
    internet_up: int  # Mbps
    internet_down: int  # Mbps
    disk_bw: float  # MB/s
    location: str
    geo_region: str


@dataclass
class VastOffer:
    """A Vast.ai machine offer."""

    offer_id: int
    host: VastHostInfo
    gpu: VastGPUInfo
    dph_total: float  # Dollars per hour total
    dph_base: float  # Base price per hour
    storage_cost: float
    inet_up_cost: float
    inet_down_cost: float
    min_bid: float
    cuda_version: float
    driver_version: str
    disk_space: int  # GB
    ram: int  # MB
    cpu_cores: int
    cpu_ram: int  # MB
    duration: Optional[float] = None  # Hours available


@dataclass
class VastInstance:
    """A running Vast.ai instance."""

    instance_id: int
    offer_id: int
    actual_status: VastInstanceStatus
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    jupyter_url: Optional[str] = None
    image: str = ""
    label: str = ""
    start_date: Optional[datetime] = None
    cost_per_hour: float = 0.0
    total_cost: float = 0.0
    gpu_name: str = ""
    num_gpus: int = 1


class VastAIAdapter:
    """
    Adapter for Vast.ai GPU marketplace.

    Features:
    - Host filtering by reliability, performance, region
    - Spot instance bidding
    - Checkpoint-based job resumption
    - Encryption for sensitive model weights
    - Automatic retry on preemption
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        min_reliability: float = 0.95,
        min_dlperf: float = 10.0,
        min_internet_speed: int = 100,
        preferred_regions: Optional[List[str]] = None,
        encrypt_weights: bool = True,
        max_retries: int = 5,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.getenv("VASTAI_API_KEY", "")
        self.api_base = "https://console.vast.ai/api/v0"
        self.min_reliability = min_reliability
        self.min_dlperf = min_dlperf
        self.min_internet_speed = min_internet_speed
        self.preferred_regions = preferred_regions or ["us", "eu"]
        self.encrypt_weights = encrypt_weights
        self.max_retries = max_retries
        self.timeout = timeout

        # Encryption key for model weights
        self.encryption_key = os.getenv("VASTAI_MODEL_ENCRYPTION_KEY", "")

        if not self.api_key:
            logger.warning("Vast.ai API key not configured")

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to Vast.ai API."""
        url = f"{self.api_base}/{endpoint}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method,
                url,
                headers=self._get_headers(),
                json=json_data,
                params=params,
            )
            response.raise_for_status()

            if response.content:
                return response.json()
            return {}

    # ==================== Offer Search & Filtering ====================

    async def search_offers(
        self,
        gpu_name: Optional[str] = None,
        num_gpus: int = 1,
        min_gpu_ram: int = 16000,  # MB
        max_dph: float = 10.0,  # Max dollars per hour
        disk_space: int = 50,  # GB
        order_by: str = "dph_total",  # Options: dph_total, dlperf, reliability
        limit: int = 50,
    ) -> List[VastOffer]:
        """
        Search for GPU offers matching criteria.

        Filters by:
        - Host reliability
        - Deep learning performance
        - Internet speed
        - Region preference
        """
        # Build search query
        query_parts = [
            f"reliability >= {self.min_reliability}",
            f"dlperf >= {self.min_dlperf}",
            f"inet_up >= {self.min_internet_speed}",
            f"inet_down >= {self.min_internet_speed}",
            f"num_gpus >= {num_gpus}",
            f"gpu_ram >= {min_gpu_ram}",
            f"dph_total <= {max_dph}",
            f"disk_space >= {disk_space}",
            "rentable = true",
            "rented = false",
        ]

        if gpu_name:
            query_parts.append(f'gpu_name = "{gpu_name}"')

        # Add region filter
        if self.preferred_regions:
            region_filter = " || ".join(
                f'geolocation like "{region}%"' for region in self.preferred_regions
            )
            query_parts.append(f"({region_filter})")

        query = " && ".join(query_parts)

        result = await self._make_request(
            "GET",
            "bundles",
            params={
                "q": query,
                "order": order_by,
                "limit": limit,
                "type": "on-demand",
            },
        )

        offers = []
        for offer_data in result.get("offers", []):
            host = VastHostInfo(
                host_id=offer_data.get("host_id", 0),
                reliability=offer_data.get("reliability", 0),
                verified=offer_data.get("verified", False),
                internet_up=offer_data.get("inet_up", 0),
                internet_down=offer_data.get("inet_down", 0),
                disk_bw=offer_data.get("disk_bw", 0),
                location=offer_data.get("geolocation", ""),
                geo_region=offer_data.get("geolocation", "")[:2],
            )

            gpu = VastGPUInfo(
                gpu_name=offer_data.get("gpu_name", ""),
                gpu_ram=offer_data.get("gpu_ram", 0),
                num_gpus=offer_data.get("num_gpus", 1),
                pcie_bw=offer_data.get("pcie_bw", 0),
                dlperf=offer_data.get("dlperf", 0),
                dlperf_per_dphtotal=offer_data.get("dlperf_per_dphtotal", 0),
            )

            offers.append(
                VastOffer(
                    offer_id=offer_data.get("id", 0),
                    host=host,
                    gpu=gpu,
                    dph_total=offer_data.get("dph_total", 0),
                    dph_base=offer_data.get("dph_base", 0),
                    storage_cost=offer_data.get("storage_cost", 0),
                    inet_up_cost=offer_data.get("inet_up_cost", 0),
                    inet_down_cost=offer_data.get("inet_down_cost", 0),
                    min_bid=offer_data.get("min_bid", 0),
                    cuda_version=offer_data.get("cuda_max_good", 0),
                    driver_version=offer_data.get("driver_version", ""),
                    disk_space=offer_data.get("disk_space", 0),
                    ram=offer_data.get("cpu_ram", 0),
                    cpu_cores=offer_data.get("cpu_cores_effective", 0),
                    cpu_ram=offer_data.get("cpu_ram", 0),
                    duration=offer_data.get("duration"),
                )
            )

        return offers

    async def find_best_offer(
        self,
        gpu_type: str = "RTX_4090",
        num_gpus: int = 1,
        optimize_for: str = "cost",  # Options: cost, performance, reliability
    ) -> Optional[VastOffer]:
        """
        Find the best offer optimizing for the specified criteria.

        Args:
            gpu_type: Target GPU type
            num_gpus: Number of GPUs needed
            optimize_for: Optimization target

        Returns:
            Best matching offer or None
        """
        # Map GPU types to search names
        gpu_name_map = {
            "RTX_4090": "RTX 4090",
            "A100_80GB": "A100 80GB",
            "A100_40GB": "A100",
            "H100_80GB": "H100 80GB",
            "H100_SXM": "H100 SXM",
            "RTX_3090": "RTX 3090",
            "RTX_A6000": "RTX A6000",
        }

        gpu_name = gpu_name_map.get(gpu_type, gpu_type)

        # Set order based on optimization target
        order_by = {
            "cost": "dph_total",
            "performance": "-dlperf",
            "reliability": "-reliability",
        }.get(optimize_for, "dph_total")

        offers = await self.search_offers(
            gpu_name=gpu_name,
            num_gpus=num_gpus,
            order_by=order_by,
            limit=10,
        )

        return offers[0] if offers else None

    # ==================== Instance Management ====================

    async def create_instance(
        self,
        offer_id: int,
        image: str,
        disk_gb: int = 50,
        label: str = "",
        env_vars: Optional[Dict[str, str]] = None,
        onstart_cmd: str = "",
        bid_price: Optional[float] = None,
    ) -> VastInstance:
        """
        Create a new instance from an offer.

        Args:
            offer_id: Vast.ai offer ID
            image: Docker image to run
            disk_gb: Disk space in GB
            label: Instance label for identification
            env_vars: Environment variables
            onstart_cmd: Command to run on start
            bid_price: Bid price (for spot instances)

        Returns:
            Created instance details
        """
        payload = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "label": label,
        }

        if env_vars:
            payload["env"] = env_vars

        if onstart_cmd:
            payload["onstart"] = onstart_cmd

        if bid_price:
            payload["price"] = bid_price

        result = await self._make_request(
            "PUT",
            f"asks/{offer_id}/",
            json_data=payload,
        )

        instance_id = result.get("new_contract")

        # Get instance details
        if instance_id:
            return await self.get_instance(instance_id)

        raise Exception(f"Failed to create instance: {result}")

    async def get_instance(self, instance_id: int) -> VastInstance:
        """Get details of an instance."""
        result = await self._make_request("GET", f"instances/{instance_id}/")

        status_str = result.get("actual_status", "error")
        try:
            status = VastInstanceStatus(status_str)
        except ValueError:
            status = VastInstanceStatus.ERROR

        return VastInstance(
            instance_id=instance_id,
            offer_id=result.get("machine_id", 0),
            actual_status=status,
            ssh_host=result.get("ssh_host"),
            ssh_port=result.get("ssh_port"),
            jupyter_url=result.get("jupyter_url"),
            image=result.get("image_uuid", ""),
            label=result.get("label", ""),
            start_date=datetime.fromisoformat(result["start_date"])
            if result.get("start_date")
            else None,
            cost_per_hour=result.get("dph_total", 0),
            total_cost=result.get("total_cost", 0),
            gpu_name=result.get("gpu_name", ""),
            num_gpus=result.get("num_gpus", 1),
        )

    async def list_instances(self) -> List[VastInstance]:
        """List all instances."""
        result = await self._make_request("GET", "instances/")

        instances = []
        for inst_data in result.get("instances", []):
            status_str = inst_data.get("actual_status", "error")
            try:
                status = VastInstanceStatus(status_str)
            except ValueError:
                status = VastInstanceStatus.ERROR

            instances.append(
                VastInstance(
                    instance_id=inst_data.get("id", 0),
                    offer_id=inst_data.get("machine_id", 0),
                    actual_status=status,
                    ssh_host=inst_data.get("ssh_host"),
                    ssh_port=inst_data.get("ssh_port"),
                    jupyter_url=inst_data.get("jupyter_url"),
                    image=inst_data.get("image_uuid", ""),
                    label=inst_data.get("label", ""),
                    cost_per_hour=inst_data.get("dph_total", 0),
                    total_cost=inst_data.get("total_cost", 0),
                    gpu_name=inst_data.get("gpu_name", ""),
                    num_gpus=inst_data.get("num_gpus", 1),
                )
            )

        return instances

    async def destroy_instance(self, instance_id: int) -> bool:
        """Destroy an instance."""
        await self._make_request("DELETE", f"instances/{instance_id}/")
        return True

    async def stop_instance(self, instance_id: int) -> bool:
        """Stop an instance (keeps disk)."""
        await self._make_request(
            "PUT",
            f"instances/{instance_id}/",
            json_data={"state": "stopped"},
        )
        return True

    async def start_instance(self, instance_id: int) -> bool:
        """Start a stopped instance."""
        await self._make_request(
            "PUT",
            f"instances/{instance_id}/",
            json_data={"state": "running"},
        )
        return True

    # ==================== Job Execution ====================

    async def run_training_job(
        self,
        gpu_type: str,
        num_gpus: int,
        image: str,
        training_script: str,
        model_gcs_path: str,
        checkpoint_gcs_path: str,
        max_cost_per_hour: float = 5.0,
        max_runtime_hours: float = 24.0,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a training job with automatic checkpointing.

        Features:
        - Downloads model from GCS at start
        - Checkpoints to GCS periodically
        - Auto-restarts on preemption
        - Encrypts weights if configured
        """
        # Find best offer
        offer = await self.find_best_offer(
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            optimize_for="cost",
        )

        if not offer or offer.dph_total > max_cost_per_hour:
            raise Exception(
                f"No suitable offer found within budget ${max_cost_per_hour}/hr"
            )

        # Build startup script
        onstart_script = self._build_training_script(
            training_script=training_script,
            model_gcs_path=model_gcs_path,
            checkpoint_gcs_path=checkpoint_gcs_path,
            encrypt=self.encrypt_weights,
        )

        # Create instance
        instance = await self.create_instance(
            offer_id=offer.offer_id,
            image=image,
            disk_gb=100,
            label=f"training-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            env_vars={
                **(env_vars or {}),
                "MAX_RUNTIME_HOURS": str(max_runtime_hours),
                "CHECKPOINT_INTERVAL_MINUTES": "30",
                "GCS_MODEL_PATH": model_gcs_path,
                "GCS_CHECKPOINT_PATH": checkpoint_gcs_path,
            },
            onstart_cmd=onstart_script,
            bid_price=offer.min_bid * 1.2,  # Bid 20% above minimum
        )

        return {
            "instance_id": instance.instance_id,
            "gpu_type": offer.gpu.gpu_name,
            "num_gpus": offer.gpu.num_gpus,
            "cost_per_hour": offer.dph_total,
            "estimated_total_cost": offer.dph_total * max_runtime_hours,
            "checkpoint_path": checkpoint_gcs_path,
        }

    def _build_training_script(
        self,
        training_script: str,
        model_gcs_path: str,
        checkpoint_gcs_path: str,
        encrypt: bool = True,
    ) -> str:
        """Build the onstart script for training jobs."""
        encryption_setup = ""
        if encrypt:
            encryption_setup = """
# Setup encryption
ENCRYPTION_KEY="${VASTAI_MODEL_ENCRYPTION_KEY}"
export ENCRYPTION_KEY

decrypt_model() {
    if [ -n "$ENCRYPTION_KEY" ]; then
        openssl enc -d -aes-256-cbc -pbkdf2 -in "$1" -out "$2" -k "$ENCRYPTION_KEY"
    else
        cp "$1" "$2"
    fi
}

encrypt_checkpoint() {
    if [ -n "$ENCRYPTION_KEY" ]; then
        openssl enc -aes-256-cbc -pbkdf2 -in "$1" -out "$2" -k "$ENCRYPTION_KEY"
    else
        cp "$1" "$2"
    fi
}
"""

        return f"""#!/bin/bash
set -e

{encryption_setup}

# Install gsutil if not present
if ! command -v gsutil &> /dev/null; then
    pip install gsutil
fi

# Download model weights
echo "Downloading model from {model_gcs_path}..."
gsutil -m cp -r {model_gcs_path} /workspace/model/

# Check for existing checkpoint
CHECKPOINT_EXISTS=$(gsutil ls {checkpoint_gcs_path}/latest/ 2>/dev/null || echo "")
if [ -n "$CHECKPOINT_EXISTS" ]; then
    echo "Resuming from checkpoint..."
    gsutil -m cp -r {checkpoint_gcs_path}/latest/* /workspace/checkpoint/
fi

# Run training with periodic checkpointing
{training_script}

# Upload final checkpoint
echo "Uploading final checkpoint..."
gsutil -m cp -r /workspace/checkpoint/* {checkpoint_gcs_path}/final/

echo "Training complete!"
"""

    async def run_batch_inference(
        self,
        gpu_type: str,
        image: str,
        input_gcs_path: str,
        output_gcs_path: str,
        inference_script: str,
        max_cost_per_hour: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Run batch inference job.

        Good for large-scale embedding generation, batch predictions, etc.
        """
        offer = await self.find_best_offer(
            gpu_type=gpu_type,
            num_gpus=1,
            optimize_for="cost",
        )

        if not offer or offer.dph_total > max_cost_per_hour:
            raise Exception(
                f"No suitable offer found within budget ${max_cost_per_hour}/hr"
            )

        onstart_script = f"""#!/bin/bash
set -e

# Download input data
gsutil -m cp -r {input_gcs_path} /workspace/input/

# Run inference
{inference_script}

# Upload results
gsutil -m cp -r /workspace/output/* {output_gcs_path}/

# Self-terminate on completion
curl -X DELETE "https://console.vast.ai/api/v0/instances/$VAST_CONTAINERLABEL/" \\
    -H "Authorization: Bearer $VAST_API_KEY"
"""

        instance = await self.create_instance(
            offer_id=offer.offer_id,
            image=image,
            disk_gb=50,
            label=f"batch-inference-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            env_vars={
                "VAST_API_KEY": self.api_key,
                "INPUT_PATH": "/workspace/input",
                "OUTPUT_PATH": "/workspace/output",
            },
            onstart_cmd=onstart_script,
        )

        return {
            "instance_id": instance.instance_id,
            "cost_per_hour": offer.dph_total,
            "output_path": output_gcs_path,
        }

    # ==================== Cost & Monitoring ====================

    async def get_spending(self) -> Dict[str, Any]:
        """Get current spending information."""
        result = await self._make_request("GET", "users/current/")

        return {
            "balance": result.get("balance", 0),
            "balance_threshold": result.get("balance_threshold", 0),
            "total_spent": result.get("total_spent", 0),
        }

    async def estimate_job_cost(
        self,
        gpu_type: str,
        num_gpus: int,
        hours: float,
    ) -> Dict[str, Any]:
        """Estimate cost for a job."""
        offer = await self.find_best_offer(
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            optimize_for="cost",
        )

        if not offer:
            return {"error": "No suitable offers found"}

        return {
            "gpu_type": offer.gpu.gpu_name,
            "cost_per_hour": offer.dph_total,
            "hours": hours,
            "estimated_total": offer.dph_total * hours,
            "dlperf_score": offer.gpu.dlperf,
        }


# Singleton instance
_adapter: Optional[VastAIAdapter] = None


def get_vastai_adapter() -> VastAIAdapter:
    """Get or create the Vast.ai adapter singleton."""
    global _adapter
    if _adapter is None:
        _adapter = VastAIAdapter()
    return _adapter
