"""
Cloud Provider Configuration for Multi-Provider Inference Platform

Manages configuration for:
- Google Cloud (Ollama + llama.cpp) - Fast dev environment, persistent model pulls
- RunPod - Production inference endpoints, multi-GPU training
- Vast.ai - Cost-sensitive batch jobs, spot H100/A100 runs

Security:
- Encryption at rest and in transit
- Signed short-lived URLs for model weights
- Treat Vast.ai hosts as untrusted for sensitive weights
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum


class CloudProvider(Enum):
    """Available cloud providers for inference/training."""

    GCP = "gcp"
    RUNPOD = "runpod"
    VASTAI = "vastai"
    LOCAL = "local"


class JobType(Enum):
    """Types of jobs for routing decisions."""

    INFERENCE_REALTIME = "inference_realtime"
    INFERENCE_BATCH = "inference_batch"
    TRAINING = "training"
    FINETUNING = "finetuning"
    PRETRAINING = "pretraining"
    EMBEDDING = "embedding"


@dataclass
class GCPConfig:
    """Google Cloud Platform configuration for Ollama/llama.cpp.

    Region Strategy (Alpha Phase):
    - Single region: us-east1 (East Coast + RunPod closest cluster)
    - Multi-region is an optimization for later when Goblin grows
    - Storage and deployment structured for easy region addition via config
    """

    project_id: str = os.getenv("GCP_PROJECT_ID", "goblin-assistant-llm")
    region: str = os.getenv("GCP_REGION", "us-east1")  # Single region for alpha
    zone: str = os.getenv("GCP_ZONE", "us-east1-b")

    # GCS bucket for model weights
    model_bucket: str = os.getenv("GCS_MODEL_BUCKET", "goblin-llm-models")
    checkpoint_bucket: str = os.getenv(
        "GCS_CHECKPOINT_BUCKET", "goblin-llm-checkpoints"
    )

    # Ollama VM configuration
    ollama_vm_name: str = os.getenv("GCP_OLLAMA_VM", "ollama-gpu")
    ollama_internal_url: str = os.getenv("OLLAMA_INTERNAL_URL", "http://10.0.0.2:11434")

    # llama.cpp Cloud Run service
    llamacpp_service_url: str = os.getenv("LLAMACPP_SERVICE_URL", "")

    # Service account for storage access
    service_account_key: str = os.getenv("GCP_SERVICE_ACCOUNT_KEY", "")

    # Signed URL configuration
    signed_url_expiry_seconds: int = 3600  # 1 hour


@dataclass
class RunPodConfig:
    """RunPod configuration for production inference and training."""

    api_key: str = os.getenv("RUNPOD_API_KEY", "")
    s3_key: str = os.getenv(
        "RUNPOD_S3_KEY", ""
    )  # Network storage for models/checkpoints
    api_base_url: str = "https://api.runpod.io/v2"

    # Serverless endpoints
    serverless_enabled: bool = True
    preferred_gpu_types: List[str] = field(
        default_factory=lambda: ["NVIDIA A100", "NVIDIA H100"]
    )

    # Instant Clusters for distributed training
    cluster_enabled: bool = True
    default_cluster_size: int = 2
    max_cluster_size: int = 8

    # Networking
    use_private_networking: bool = True

    # Network Volume Storage
    network_volume_enabled: bool = True
    network_volume_region: str = os.getenv("RUNPOD_VOLUME_REGION", "US")

    # Default container image
    default_image: str = "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel"

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 30

    # Cost limits
    max_hourly_cost: float = 10.0  # USD per hour

    # Model storage integration
    use_gcs_for_models: bool = True


@dataclass
class VastAIConfig:
    """Vast.ai configuration for cost-sensitive batch jobs.

    Budget Range (Dev/Alpha): $50-200/month
    Strategy:
    - Spot 4090/3090 only (low uptime is acceptable)
    - Dev/experimental jobs only
    - Checkpoint frequently (preemption risk)
    - Treat all hosts as untrusted
    """

    api_key: str = os.getenv("VASTAI_API_KEY", "")
    api_base_url: str = "https://console.vast.ai/api/v0"

    # Budget constraints ($50-200/month)
    monthly_budget_usd: float = float(os.getenv("VASTAI_MONTHLY_BUDGET", "100"))
    max_cost_per_hour: float = float(
        os.getenv("VASTAI_MAX_COST_PER_HOUR", "0.80")
    )  # ~$0.40-0.80/hr for spot 4090

    # Host filtering criteria (relaxed for budget)
    min_host_reliability: float = 0.90  # 90% uptime acceptable for dev
    min_host_dlperf: float = 8.0  # Lower threshold for budget GPUs
    min_internet_speed: int = 50  # Mbps (relaxed)
    preferred_regions: List[str] = field(
        default_factory=lambda: ["us"]
    )  # US only for latency

    # GPU preferences (budget-focused: spot 4090/3090)
    preferred_gpu_types: List[str] = field(
        default_factory=lambda: [
            "RTX_4090",  # Best price/perf for inference (~$0.40-0.60/hr spot)
            "RTX_3090",  # Budget option (~$0.25-0.40/hr spot)
            "RTX_3090_Ti",  # Slightly better than 3090
        ]
    )

    # Excluded for budget mode (too expensive)
    excluded_gpu_types: List[str] = field(
        default_factory=lambda: [
            "A100_80GB",  # $1.50+/hr
            "H100_80GB",  # $2.50+/hr
            "H100_SXM",  # $3.00+/hr
        ]
    )

    # Quantization preferences
    prefer_quantized: bool = True  # Use GGUF/4-bit when possible

    # Spot instance configuration (aggressive for budget)
    use_spot_instances: bool = True
    spot_bid_multiplier: float = 1.0  # Bid at spot price (accept preemption)
    allow_interruptible: bool = True  # Accept interruptible instances

    # Security - treat hosts as untrusted
    encrypt_model_weights: bool = True
    encryption_key_env: str = "VASTAI_MODEL_ENCRYPTION_KEY"

    # Checkpoint configuration (frequent for preemption resilience)
    checkpoint_interval_minutes: int = 15  # More frequent for spot
    checkpoint_to_gcs: bool = True

    # Retry and resilience
    max_retries: int = 10  # More retries for spot preemption
    retry_delay_seconds: int = 30  # Faster retry
    auto_restart_on_preemption: bool = True

    # Job limits
    max_concurrent_jobs: int = 1  # Budget mode: one job at a time
    max_runtime_hours: int = 8  # Limit job duration for cost control


@dataclass
class ModelStorageConfig:
    """Configuration for model weight storage and distribution."""

    # Primary storage (GCS)
    primary_bucket: str = os.getenv("GCS_MODEL_BUCKET", "goblin-llm-models")

    # Fallback storage (S3)
    s3_bucket: str = os.getenv("S3_MODEL_BUCKET", "")
    s3_region: str = os.getenv("S3_REGION", "us-east-1")

    # Model download settings
    download_timeout_seconds: int = 1800  # 30 minutes for large models
    verify_checksum: bool = True

    # Quantization paths
    quantized_model_suffix: str = "_quantized"
    gguf_model_suffix: str = ".gguf"

    # Encryption
    encrypt_at_rest: bool = True
    encrypt_in_transit: bool = True


@dataclass
class CloudProvidersConfig:
    """Master configuration for all cloud providers."""

    gcp: GCPConfig = field(default_factory=GCPConfig)
    runpod: RunPodConfig = field(default_factory=RunPodConfig)
    vastai: VastAIConfig = field(default_factory=VastAIConfig)
    storage: ModelStorageConfig = field(default_factory=ModelStorageConfig)

    # Provider priority for different job types
    provider_priority: Dict[JobType, List[CloudProvider]] = field(
        default_factory=lambda: {
            JobType.INFERENCE_REALTIME: [
                CloudProvider.GCP,
                CloudProvider.RUNPOD,
                CloudProvider.LOCAL,
            ],
            JobType.INFERENCE_BATCH: [
                CloudProvider.VASTAI,
                CloudProvider.RUNPOD,
                CloudProvider.GCP,
            ],
            JobType.TRAINING: [CloudProvider.RUNPOD, CloudProvider.VASTAI],
            JobType.FINETUNING: [CloudProvider.RUNPOD, CloudProvider.VASTAI],
            JobType.PRETRAINING: [CloudProvider.VASTAI, CloudProvider.RUNPOD],
            JobType.EMBEDDING: [
                CloudProvider.GCP,
                CloudProvider.LOCAL,
                CloudProvider.RUNPOD,
            ],
        }
    )

    # Cost thresholds for automatic routing
    cost_sensitive_threshold: float = 1.0  # Jobs over $1/hour go to Vast.ai
    latency_sensitive_ms: int = 100  # Jobs needing <100ms go to GCP/local


# Global configuration instance
_config: Optional[CloudProvidersConfig] = None


def get_cloud_config() -> CloudProvidersConfig:
    """Get or create the global cloud providers configuration."""
    global _config
    if _config is None:
        _config = CloudProvidersConfig()
    return _config


def get_provider_for_job(
    job_type: JobType,
    cost_estimate: float = 0.0,
    latency_requirement_ms: Optional[int] = None,
) -> CloudProvider:
    """Select the best provider for a given job type and requirements."""
    config = get_cloud_config()

    # Check latency requirements
    if latency_requirement_ms and latency_requirement_ms < config.latency_sensitive_ms:
        # Need low latency - prefer GCP or local
        return CloudProvider.GCP

    # Check cost sensitivity
    if cost_estimate > config.cost_sensitive_threshold:
        # Cost-sensitive - prefer Vast.ai
        if config.vastai.api_key:
            return CloudProvider.VASTAI

    # Use default priority
    priority = config.provider_priority.get(job_type, [CloudProvider.GCP])

    # Return first available provider
    for provider in priority:
        if _is_provider_available(provider, config):
            return provider

    # Fallback to local
    return CloudProvider.LOCAL


def _is_provider_available(
    provider: CloudProvider, config: CloudProvidersConfig
) -> bool:
    """Check if a provider is configured and available."""
    if provider == CloudProvider.GCP:
        return bool(config.gcp.project_id)
    elif provider == CloudProvider.RUNPOD:
        return bool(config.runpod.api_key)
    elif provider == CloudProvider.VASTAI:
        return bool(config.vastai.api_key)
    elif provider == CloudProvider.LOCAL:
        return True
    return False
