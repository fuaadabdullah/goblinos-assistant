"""
Base Provider Interface
Abstract base class for all cloud providers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ProviderType(str, Enum):
    """Supported cloud providers."""
    GCP = "gcp"
    RUNPOD = "runpod"
    VASTAI = "vastai"


class JobType(str, Enum):
    """Types of jobs that can be routed."""
    INFERENCE = "inference"           # Real-time inference
    BATCH_INFERENCE = "batch_inference"  # Batch processing
    TRAINING = "training"             # Model training
    FINE_TUNING = "fine_tuning"      # Fine-tuning/LoRA
    SWEEP = "sweep"                   # Hyperparameter sweeps


class GPUType(str, Enum):
    """Common GPU types across providers."""
    RTX_3090 = "rtx_3090"
    RTX_4090 = "rtx_4090"
    A100_40GB = "a100_40gb"
    A100_80GB = "a100_80gb"
    H100 = "h100"


@dataclass
class JobResult:
    """Result from a provider job."""
    success: bool
    provider: ProviderType
    job_id: str
    output: Any
    cost: float
    duration_seconds: float
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProviderCapabilities:
    """Capabilities of a provider."""
    supports_inference: bool = True
    supports_training: bool = False
    supports_multi_gpu: bool = False
    max_gpus: int = 1
    available_gpu_types: Optional[list[GPUType]] = None
    supports_spot: bool = False
    supports_streaming: bool = False
    
    def __post_init__(self) -> None:
        if self.available_gpu_types is None:
            self.available_gpu_types = []


class BaseProvider(ABC):
    """
    Abstract base class for cloud providers.
    
    Each provider implementation must:
    - Report health and availability
    - Estimate costs and latency
    - Submit and manage jobs
    - Handle authentication
    """
    
    provider_type: ProviderType
    capabilities: ProviderCapabilities
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and reachable."""
        pass
    
    @abstractmethod
    async def get_availability(self) -> float:
        """Get availability score (0-100)."""
        pass
    
    @abstractmethod
    async def get_cost_estimate(
        self, 
        job_type: JobType,
        gpu_type: Optional[str] = None,
    ) -> float:
        """Get estimated cost per hour for job type."""
        pass
    
    @abstractmethod
    async def get_latency_estimate(self, job_type: JobType) -> float:
        """Get estimated latency in milliseconds."""
        pass
    
    def supports_job_type(self, job_type: JobType) -> bool:
        """Check if provider supports the job type."""
        if job_type in [JobType.INFERENCE, JobType.BATCH_INFERENCE]:
            return self.capabilities.supports_inference
        elif job_type in [JobType.TRAINING, JobType.FINE_TUNING, JobType.SWEEP]:
            return self.capabilities.supports_training
        return False
    
    def has_gpu(self, gpu_type: str) -> bool:
        """Check if provider has the specified GPU type."""
        if self.capabilities.available_gpu_types is None:
            return False
        try:
            gpu = GPUType(gpu_type.lower().replace(" ", "_").replace("-", "_"))
            return gpu in self.capabilities.available_gpu_types
        except ValueError:
            return False
    
    @abstractmethod
    async def submit_inference(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> JobResult:
        """Submit an inference request."""
        pass
    
    @abstractmethod
    async def submit_training(
        self,
        config: dict[str, Any],
        checkpoint_path: Optional[str] = None,
    ) -> JobResult:
        """Submit a training job."""
        pass
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a submitted job."""
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        pass
