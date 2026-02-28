"""
Multi-Provider Inference Orchestrator Router

Provides API endpoints for multi-cloud inference routing and management.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])


# =============================================================================
# Request/Response Models
# =============================================================================


class InferenceRequest(BaseModel):
    """Request model for inference."""

    prompt: str = Field(..., description="The prompt to send to the model")
    model: Optional[str] = Field(None, description="Specific model to use")
    provider: Optional[str] = Field(None, description="Specific provider to use")
    routing_strategy: Optional[str] = Field(
        None,
        description="Routing strategy: LOWEST_LATENCY, LOWEST_COST, HIGHEST_QUALITY, ROUND_ROBIN, WEIGHTED",
    )
    max_tokens: Optional[int] = Field(2048, description="Maximum tokens in response")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    timeout: Optional[int] = Field(60, description="Request timeout in seconds")


class InferenceResponse(BaseModel):
    """Response model for inference."""

    text: str
    model: str
    provider: str
    tokens_used: int
    latency_ms: float
    cost_estimate: Optional[float] = None
    cached: bool = False


class ProviderHealth(BaseModel):
    """Provider health status."""

    provider: str
    status: str  # healthy, degraded, unhealthy
    latency_ms: Optional[float] = None
    error_rate: Optional[float] = None
    last_check: Optional[str] = None


class HealthResponse(BaseModel):
    """Overall health response."""

    status: str
    providers: List[ProviderHealth]
    active_provider_count: int
    total_provider_count: int


class TrainingJobRequest(BaseModel):
    """Request to start a training job."""

    model_name: str = Field(..., description="Name for the trained model")
    base_model: str = Field(..., description="Base model to fine-tune")
    dataset_gcs_path: str = Field(..., description="GCS path to training dataset")
    provider: str = Field("vastai", description="Cloud provider: runpod, vastai, gcp")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Training configuration"
    )


class TrainingJobResponse(BaseModel):
    """Response for training job."""

    job_id: str
    status: str
    provider: str
    estimated_cost: Optional[float] = None


# =============================================================================
# Lazy Initialization
# =============================================================================

_orchestrator = None
_model_registry = None


def get_orchestrator():
    """Lazily initialize the orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        try:
            from ..services.inference_orchestrator import (
                MultiProviderOrchestrator,
                RoutingStrategy,
            )
            from ..config.cloud_providers import (
                CloudProviderConfig,
                GCPConfig,
                RunPodConfig,
                VastAIConfig,
            )

            # Load config from environment
            config = CloudProviderConfig(
                gcp=GCPConfig(
                    project_id=os.getenv("GCS_PROJECT_ID", "goblin-assistant-llm"),
                    region=os.getenv("GCP_REGION", "us-east1"),
                ),
                runpod=RunPodConfig(
                    api_key=os.getenv("RUNPOD_API_KEY", ""),
                ),
                vastai=VastAIConfig(
                    api_key=os.getenv("VASTAI_API_KEY", ""),
                ),
            )

            _orchestrator = MultiProviderOrchestrator(config)
            logger.info("Initialized MultiProviderOrchestrator")
        except ImportError as e:
            logger.warning(f"Could not initialize orchestrator: {e}")
            return None
        except Exception as e:
            logger.error(f"Error initializing orchestrator: {e}")
            return None
    return _orchestrator


def get_model_registry():
    """Lazily initialize the model registry."""
    global _model_registry
    if _model_registry is None:
        try:
            from ..services.model_registry import ModelRegistry

            _model_registry = ModelRegistry()
            logger.info("Initialized ModelRegistry")
        except ImportError as e:
            logger.warning(f"Could not initialize model registry: {e}")
            return None
        except Exception as e:
            logger.error(f"Error initializing model registry: {e}")
            return None
    return _model_registry


# =============================================================================
# Inference Endpoints
# =============================================================================


@router.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """
    Route inference request to the optimal provider.

    Uses the configured routing strategy to select the best provider
    based on latency, cost, quality, or other criteria.
    """
    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Inference orchestrator not available. Check cloud provider configuration.",
        )

    try:
        # Parse routing strategy
        routing_strategy = None
        if request.routing_strategy:
            try:
                from ..services.inference_orchestrator import RoutingStrategy

                routing_strategy = RoutingStrategy[request.routing_strategy.upper()]
            except (KeyError, AttributeError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid routing strategy: {request.routing_strategy}",
                )

        result = await orchestrator.route_inference(
            prompt=request.prompt,
            model=request.model,
            provider=request.provider,
            strategy=routing_strategy,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt,
            timeout=request.timeout,
        )

        return InferenceResponse(
            text=result.get("text", ""),
            model=result.get("model", "unknown"),
            provider=result.get("provider", "unknown"),
            tokens_used=result.get("tokens_used", 0),
            latency_ms=result.get("latency_ms", 0),
            cost_estimate=result.get("cost_estimate"),
            cached=result.get("cached", False),
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def check_health():
    """
    Check health of all configured providers.

    Returns status of each provider including latency and error rates.
    """
    orchestrator = get_orchestrator()
    if orchestrator is None:
        return HealthResponse(
            status="degraded",
            providers=[],
            active_provider_count=0,
            total_provider_count=0,
        )

    try:
        health_data = await orchestrator.check_all_providers()

        providers = []
        for name, status in health_data.get("providers", {}).items():
            providers.append(
                ProviderHealth(
                    provider=name,
                    status=status.get("status", "unknown"),
                    latency_ms=status.get("latency_ms"),
                    error_rate=status.get("error_rate"),
                    last_check=status.get("last_check"),
                )
            )

        active_count = sum(1 for p in providers if p.status == "healthy")

        overall_status = "healthy"
        if active_count == 0:
            overall_status = "unhealthy"
        elif active_count < len(providers):
            overall_status = "degraded"

        return HealthResponse(
            status=overall_status,
            providers=providers,
            active_provider_count=active_count,
            total_provider_count=len(providers),
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="error",
            providers=[],
            active_provider_count=0,
            total_provider_count=0,
        )


@router.get("/providers")
async def list_providers():
    """List all configured providers and their capabilities."""
    orchestrator = get_orchestrator()
    if orchestrator is None:
        return {"providers": [], "message": "Orchestrator not initialized"}

    try:
        return await orchestrator.list_providers()
    except Exception as e:
        logger.error(f"Error listing providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models(provider: Optional[str] = Query(None)):
    """List available models, optionally filtered by provider."""
    orchestrator = get_orchestrator()
    if orchestrator is None:
        return {"models": [], "message": "Orchestrator not initialized"}

    try:
        return await orchestrator.list_models(provider=provider)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Training Endpoints
# =============================================================================


@router.post("/training/start", response_model=TrainingJobResponse)
async def start_training(
    request: TrainingJobRequest, background_tasks: BackgroundTasks
):
    """
    Start a model training job on a cloud provider.

    Supports RunPod, Vast.ai, and GCP for training infrastructure.
    """
    try:
        from tasks.model_training_worker import train_model_task

        # Submit training job to Celery
        task = train_model_task.delay(
            model_name=request.model_name,
            base_model=request.base_model,
            dataset_path=request.dataset_gcs_path,
            provider=request.provider,
            config=request.config,
        )

        return TrainingJobResponse(
            job_id=task.id,
            status="submitted",
            provider=request.provider,
            estimated_cost=_estimate_training_cost(request),
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Training worker not available. Install cloud dependencies.",
        )
    except Exception as e:
        logger.error(f"Training job error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/{job_id}")
async def get_training_status(job_id: str):
    """Get status of a training job."""
    try:
        from celery.result import AsyncResult
        from celery_app import app

        result = AsyncResult(job_id, app=app)

        return {
            "job_id": job_id,
            "status": result.status,
            "result": result.result if result.successful() else None,
            "error": str(result.result) if result.failed() else None,
        }

    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/training/{job_id}")
async def cancel_training(job_id: str):
    """Cancel a running training job."""
    try:
        from celery.result import AsyncResult
        from celery_app import app

        result = AsyncResult(job_id, app=app)
        result.revoke(terminate=True)

        return {"job_id": job_id, "status": "cancelled"}

    except Exception as e:
        logger.error(f"Error cancelling training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Model Registry Endpoints
# =============================================================================


@router.get("/registry/models")
async def list_registered_models(
    stage: Optional[str] = Query(
        None, description="Filter by stage: staging, production, archived"
    ),
):
    """List all registered models."""
    registry = get_model_registry()
    if registry is None:
        return {"models": [], "message": "Model registry not initialized"}

    try:
        models = await registry.list_models(stage=stage)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/models/{model_name}")
async def get_model_info(model_name: str, version: Optional[str] = Query(None)):
    """Get information about a specific model."""
    registry = get_model_registry()
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")

    try:
        model = await registry.get_model(model_name, version=version)
        if model is None:
            raise HTTPException(
                status_code=404, detail=f"Model not found: {model_name}"
            )
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/registry/models/{model_name}/promote")
async def promote_model(model_name: str, version: str = Query(...)):
    """Promote a model version to production."""
    registry = get_model_registry()
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")

    try:
        result = await registry.promote_to_production(model_name, version)
        return {"status": "promoted", "model": model_name, "version": version, **result}
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/registry/models/{model_name}/rollback")
async def rollback_model(model_name: str):
    """Rollback to the previous production version."""
    registry = get_model_registry()
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")

    try:
        result = await registry.rollback(model_name)
        return {"status": "rolled_back", "model": model_name, **result}
    except Exception as e:
        logger.error(f"Error rolling back model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Helper Functions
# =============================================================================


def _estimate_training_cost(request: TrainingJobRequest) -> Optional[float]:
    """Estimate the cost of a training job."""
    # Simple cost estimation based on provider
    cost_per_hour = {
        "runpod": 0.74,  # A100 80GB
        "vastai": 0.80,  # Max price in budget mode
        "gcp": 2.95,  # A100 40GB
    }

    # Estimate training time based on model size
    base_hours = {
        "llama-7b": 4,
        "llama-13b": 8,
        "mistral-7b": 4,
        "default": 6,
    }

    provider_cost = cost_per_hour.get(request.provider, 1.0)
    model_hours = base_hours.get(request.base_model, base_hours["default"])

    return round(provider_cost * model_hours, 2)
