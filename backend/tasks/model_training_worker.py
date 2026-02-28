"""
Model Training Worker for Celery

Handles distributed training jobs across:
- RunPod Instant Clusters
- Vast.ai spot instances
- GCP GPU VMs

Features:
- Automatic checkpointing to GCS
- Resumable training jobs
- Multi-GPU distributed training
- Cost tracking and limits
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from celery import shared_task

from ..config.cloud_providers import (
    get_cloud_config,
    CloudProvider,
    JobType,
    get_provider_for_job,
)
from ..providers.runpod_adapter import get_runpod_adapter
from ..providers.vastai_adapter import get_vastai_adapter
from ..services.model_storage import get_model_storage, QuantizationType

logger = logging.getLogger(__name__)


# ==================== Training Job Management ====================


@shared_task(
    bind=True,
    queue="batch",
    max_retries=3,
    default_retry_delay=300,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def submit_training_job(
    self,
    job_id: str,
    model_name: str,
    base_model_id: str,
    training_config: Dict[str, Any],
    provider: Optional[str] = None,
    max_cost: float = 100.0,
    max_runtime_hours: float = 24.0,
) -> Dict[str, Any]:
    """
    Submit a training job to the appropriate cloud provider.

    Args:
        job_id: Unique job identifier
        model_name: Name for the trained model
        base_model_id: ID of the base model to fine-tune
        training_config: Training hyperparameters and settings
        provider: Force specific provider (runpod, vastai, gcp)
        max_cost: Maximum total cost in USD
        max_runtime_hours: Maximum runtime in hours

    Returns:
        Job status and details
    """
    config = get_cloud_config()

    # Select provider based on job type and cost
    estimated_cost_per_hour = _estimate_hourly_cost(training_config)

    if provider:
        selected_provider = CloudProvider(provider)
    else:
        selected_provider = get_provider_for_job(
            JobType.FINETUNING if "fine" in model_name.lower() else JobType.TRAINING,
            cost_estimate=estimated_cost_per_hour,
        )

    logger.info(f"Submitting training job {job_id} to {selected_provider.value}")

    # Run async job submission
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        if selected_provider == CloudProvider.RUNPOD:
            result = loop.run_until_complete(
                _submit_runpod_training(
                    job_id,
                    model_name,
                    base_model_id,
                    training_config,
                    max_cost,
                    max_runtime_hours,
                )
            )
        elif selected_provider == CloudProvider.VASTAI:
            result = loop.run_until_complete(
                _submit_vastai_training(
                    job_id,
                    model_name,
                    base_model_id,
                    training_config,
                    max_cost,
                    max_runtime_hours,
                )
            )
        else:
            raise ValueError(f"Unsupported provider: {selected_provider}")

        return {
            "job_id": job_id,
            "provider": selected_provider.value,
            "status": "submitted",
            **result,
        }
    finally:
        loop.close()


async def _submit_runpod_training(
    job_id: str,
    model_name: str,
    base_model_id: str,
    training_config: Dict[str, Any],
    max_cost: float,
    max_runtime_hours: float,
) -> Dict[str, Any]:
    """Submit training job to RunPod."""
    adapter = get_runpod_adapter()
    storage = get_model_storage()

    # Get base model URL
    model_metadata = await storage.get_model_metadata(base_model_id)
    if not model_metadata:
        raise ValueError(f"Base model {base_model_id} not found")

    # Generate signed URL for model download
    signed_url = await storage.generate_signed_url(base_model_id)

    # Determine cluster size based on model size
    num_gpus = training_config.get("num_gpus", 1)
    gpu_type = training_config.get("gpu_type", "NVIDIA A100 80GB")

    if num_gpus > 1:
        # Create Instant Cluster
        cluster = await adapter.create_cluster(
            name=f"training-{job_id}",
            gpu_type=gpu_type,
            gpu_count_per_node=1,
            node_count=num_gpus,
            image=training_config.get(
                "image", "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel"
            ),
            volume_size_gb=100,
            env_vars={
                "MODEL_URL": signed_url.url,
                "MODEL_CHECKSUM": signed_url.checksum,
                "JOB_ID": job_id,
                "GCS_CHECKPOINT_PATH": f"gs://{storage.checkpoint_bucket}/checkpoints/{job_id}/",
                "TRAINING_CONFIG": str(training_config),
                "MAX_RUNTIME_HOURS": str(max_runtime_hours),
            },
        )

        return {
            "cluster_name": cluster["cluster_name"],
            "total_gpus": cluster["total_gpus"],
            "pods": cluster["pods"],
            "estimated_cost_per_hour": _estimate_runpod_cost(gpu_type, num_gpus),
        }
    else:
        # Single GPU pod
        pod = await adapter.create_pod(
            name=f"training-{job_id}",
            image=training_config.get(
                "image", "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel"
            ),
            gpu_type=gpu_type,
            gpu_count=1,
            volume_size_gb=100,
            env_vars={
                "MODEL_URL": signed_url.url,
                "MODEL_CHECKSUM": signed_url.checksum,
                "JOB_ID": job_id,
                "GCS_CHECKPOINT_PATH": f"gs://{storage.checkpoint_bucket}/checkpoints/{job_id}/",
            },
        )

        return {
            "pod_id": pod.pod_id,
            "gpu_type": pod.gpu_type,
            "cost_per_hour": pod.cost_per_hour,
        }


async def _submit_vastai_training(
    job_id: str,
    model_name: str,
    base_model_id: str,
    training_config: Dict[str, Any],
    max_cost: float,
    max_runtime_hours: float,
) -> Dict[str, Any]:
    """Submit training job to Vast.ai."""
    adapter = get_vastai_adapter()
    storage = get_model_storage()

    # Get base model path
    model_metadata = await storage.get_model_metadata(base_model_id)
    if not model_metadata:
        raise ValueError(f"Base model {base_model_id} not found")

    gpu_type = training_config.get("gpu_type", "A100_80GB")
    num_gpus = training_config.get("num_gpus", 1)

    # Build training script
    training_script = training_config.get("script", "python train.py")

    result = await adapter.run_training_job(
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        image=training_config.get(
            "image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel"
        ),
        training_script=training_script,
        model_gcs_path=model_metadata.gcs_path,
        checkpoint_gcs_path=f"gs://{storage.checkpoint_bucket}/checkpoints/{job_id}",
        max_cost_per_hour=max_cost / max_runtime_hours,
        max_runtime_hours=max_runtime_hours,
        env_vars={
            "JOB_ID": job_id,
            "MODEL_NAME": model_name,
            **training_config.get("env_vars", {}),
        },
    )

    return result


def _estimate_hourly_cost(training_config: Dict[str, Any]) -> float:
    """Estimate hourly cost based on training config."""
    gpu_type = training_config.get("gpu_type", "A100_80GB")
    num_gpus = training_config.get("num_gpus", 1)

    # Approximate costs per hour per GPU
    costs = {
        "RTX_4090": 0.44,
        "A100_40GB": 1.10,
        "A100_80GB": 1.59,
        "H100_80GB": 2.49,
    }

    base_cost = costs.get(gpu_type, 1.0)
    return base_cost * num_gpus


def _estimate_runpod_cost(gpu_type: str, num_gpus: int) -> float:
    """Estimate RunPod hourly cost."""
    costs = {
        "NVIDIA GeForce RTX 4090": 0.44,
        "NVIDIA A100 80GB": 1.99,
        "NVIDIA H100 80GB": 3.99,
    }
    base_cost = costs.get(gpu_type, 1.0)
    return base_cost * num_gpus


# ==================== Checkpoint Management ====================


@shared_task(queue="batch")
def save_checkpoint(
    job_id: str,
    checkpoint_path: str,
    checkpoint_name: str = "latest",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Save a training checkpoint to GCS.

    Args:
        job_id: Training job ID
        checkpoint_path: Local path to checkpoint files
        checkpoint_name: Name for this checkpoint
        metadata: Additional metadata

    Returns:
        Checkpoint details
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        storage = get_model_storage()
        gcs_path = loop.run_until_complete(
            storage.save_checkpoint(
                job_id=job_id,
                local_path=checkpoint_path,
                checkpoint_name=checkpoint_name,
                metadata=metadata,
            )
        )

        return {
            "job_id": job_id,
            "checkpoint_name": checkpoint_name,
            "gcs_path": gcs_path,
            "timestamp": datetime.utcnow().isoformat(),
        }
    finally:
        loop.close()


@shared_task(queue="batch")
def restore_checkpoint(
    job_id: str,
    checkpoint_name: str = "latest",
    local_path: str = "/workspace/checkpoint",
) -> Dict[str, Any]:
    """
    Restore a checkpoint from GCS.

    Args:
        job_id: Training job ID
        checkpoint_name: Checkpoint to restore
        local_path: Where to restore files

    Returns:
        Restore details
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        storage = get_model_storage()

        # Get checkpoint path
        checkpoint_gcs = loop.run_until_complete(storage.get_latest_checkpoint(job_id))

        if not checkpoint_gcs:
            return {
                "job_id": job_id,
                "status": "no_checkpoint",
                "message": "No checkpoint found for this job",
            }

        # Download would be handled by gsutil in the container
        return {
            "job_id": job_id,
            "checkpoint_name": checkpoint_name,
            "gcs_path": checkpoint_gcs,
            "local_path": local_path,
            "status": "ready",
        }
    finally:
        loop.close()


# ==================== Job Monitoring ====================


@shared_task(queue="high_priority")
def check_training_job_status(
    job_id: str,
    provider: str,
    resource_id: str,
) -> Dict[str, Any]:
    """
    Check the status of a training job.

    Args:
        job_id: Training job ID
        provider: Cloud provider (runpod, vastai)
        resource_id: Provider-specific resource ID

    Returns:
        Job status
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        if provider == "runpod":
            adapter = get_runpod_adapter()
            pods = loop.run_until_complete(adapter.list_pods())

            # Find our pod
            for pod in pods:
                if pod.pod_id == resource_id:
                    return {
                        "job_id": job_id,
                        "provider": provider,
                        "status": pod.status,
                        "cost_per_hour": pod.cost_per_hour,
                    }

            return {
                "job_id": job_id,
                "provider": provider,
                "status": "not_found",
            }

        elif provider == "vastai":
            adapter = get_vastai_adapter()
            instance = loop.run_until_complete(adapter.get_instance(int(resource_id)))

            return {
                "job_id": job_id,
                "provider": provider,
                "status": instance.actual_status.value,
                "cost_per_hour": instance.cost_per_hour,
                "total_cost": instance.total_cost,
            }

        else:
            return {
                "job_id": job_id,
                "provider": provider,
                "status": "unknown_provider",
            }
    finally:
        loop.close()


@shared_task(queue="high_priority")
def terminate_training_job(
    job_id: str,
    provider: str,
    resource_id: str,
    save_final_checkpoint: bool = True,
) -> Dict[str, Any]:
    """
    Terminate a training job.

    Args:
        job_id: Training job ID
        provider: Cloud provider
        resource_id: Provider-specific resource ID
        save_final_checkpoint: Whether to save final checkpoint

    Returns:
        Termination status
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        if provider == "runpod":
            adapter = get_runpod_adapter()

            # Check if it's a cluster
            if resource_id.startswith("training-"):
                terminated = loop.run_until_complete(
                    adapter.terminate_cluster(resource_id)
                )
                return {
                    "job_id": job_id,
                    "status": "terminated",
                    "nodes_terminated": terminated,
                }
            else:
                loop.run_until_complete(adapter.terminate_pod(resource_id))
                return {
                    "job_id": job_id,
                    "status": "terminated",
                }

        elif provider == "vastai":
            adapter = get_vastai_adapter()
            loop.run_until_complete(adapter.destroy_instance(int(resource_id)))
            return {
                "job_id": job_id,
                "status": "terminated",
            }

        else:
            return {
                "job_id": job_id,
                "status": "unknown_provider",
            }
    finally:
        loop.close()


# ==================== Model Export ====================


@shared_task(queue="batch")
def export_trained_model(
    job_id: str,
    model_name: str,
    version: str,
    checkpoint_name: str = "final",
    quantize: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export a trained model from checkpoint to the model registry.

    Args:
        job_id: Training job ID
        model_name: Name for the exported model
        version: Version string
        checkpoint_name: Which checkpoint to export
        quantize: Quantization type (e.g., "q4_k_m", "q8_0")

    Returns:
        Export status and model metadata
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        storage = get_model_storage()

        # Get checkpoint location
        checkpoints = loop.run_until_complete(storage.list_checkpoints(job_id))

        checkpoint = next(
            (c for c in checkpoints if c["name"] == checkpoint_name), None
        )

        if not checkpoint:
            return {
                "status": "error",
                "message": f"Checkpoint {checkpoint_name} not found",
            }

        # Determine quantization type
        quant_type = QuantizationType.NONE
        if quantize:
            quant_map = {
                "q4_k_m": QuantizationType.GGUF_Q4_K_M,
                "q5_k_m": QuantizationType.GGUF_Q5_K_M,
                "q8_0": QuantizationType.GGUF_Q8_0,
                "int8": QuantizationType.INT8,
                "int4": QuantizationType.INT4,
            }
            quant_type = quant_map.get(quantize.lower(), QuantizationType.NONE)

        # Note: Actual export would involve downloading checkpoint,
        # potentially quantizing, and uploading to model registry
        # This is a placeholder that returns the expected structure

        return {
            "status": "queued",
            "job_id": job_id,
            "model_name": model_name,
            "version": version,
            "checkpoint": checkpoint_name,
            "quantization": quant_type.value,
            "message": "Model export queued for processing",
        }
    finally:
        loop.close()
