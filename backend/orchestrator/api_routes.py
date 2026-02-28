"""
Multi-Cloud Orchestrator API Routes
Provides endpoints for routing inference across GCP, RunPod, and Vast.ai
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from .router import ProviderRouter, RoutingStrategy, get_router
from .providers.base import ProviderType, JobType
from .cost_optimizer import CostOptimizer
from .health import HealthMonitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])

# Global instances
_cost_optimizer: Optional[CostOptimizer] = None
_health_monitor: Optional[HealthMonitor] = None


def get_cost_optimizer() -> CostOptimizer:
    global _cost_optimizer
    if _cost_optimizer is None:
        _cost_optimizer = CostOptimizer()
    return _cost_optimizer


def get_health_monitor() -> HealthMonitor:
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


# Request/Response Models
class InferenceRequest(BaseModel):
    """Request for routed inference"""
    messages: list[dict] = Field(..., description="Chat messages")
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    strategy: str = Field("balanced", description="Routing strategy")
    prefer_provider: Optional[str] = Field(None, description="Preferred provider")
    max_cost: Optional[float] = Field(None, description="Max cost in USD")
    max_latency_ms: Optional[float] = Field(None, description="Max latency in ms")


class InferenceResponse(BaseModel):
    """Response from routed inference"""
    id: str
    provider: str
    model: str
    content: str
    usage: dict
    cost_estimate: float
    latency_ms: float


class RoutingRequest(BaseModel):
    """Request for routing decision"""
    job_type: str = Field("inference", description="Job type")
    strategy: str = Field("balanced", description="Routing strategy") 
    required_gpu: Optional[str] = None
    max_cost_per_hour: Optional[float] = None
    max_latency_ms: Optional[float] = None
    prefer_provider: Optional[str] = None


class RoutingResponse(BaseModel):
    """Routing decision response"""
    provider: str
    reason: str
    estimated_cost: float
    estimated_latency_ms: float
    fallback_provider: Optional[str]
    scores: dict


class CostEstimateRequest(BaseModel):
    """Request for cost estimation"""
    provider: str
    job_type: str = "inference"
    input_tokens: int
    output_tokens: int
    gpu_type: str = "default"


class CostEstimateResponse(BaseModel):
    """Cost estimation response"""
    provider: str
    estimated_cost_usd: float
    estimated_duration_seconds: float
    confidence: float
    breakdown: dict


# Routes
@router.get("/health")
async def orchestrator_health():
    """Get orchestrator health status"""
    try:
        provider_router = get_router()
        health = await provider_router.get_all_health()
        
        return {
            "status": "healthy",
            "providers": health,
            "available_providers": [
                p for p, h in health.items() if h
            ]
        }
    except Exception as e:
        logger.error(f"Orchestrator health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e)
        }


@router.post("/route", response_model=RoutingResponse)
async def get_routing_decision(request: RoutingRequest):
    """Get optimal provider routing decision"""
    try:
        provider_router = get_router()
        
        # Parse strategy
        strategy = RoutingStrategy(request.strategy)
        job_type = JobType(request.job_type)
        prefer_provider = ProviderType(request.prefer_provider) if request.prefer_provider else None
        
        decision = await provider_router.route(
            job_type=job_type,
            strategy=strategy,
            required_gpu=request.required_gpu,
            max_cost_per_hour=request.max_cost_per_hour,
            max_latency_ms=request.max_latency_ms,
            prefer_provider=prefer_provider,
        )
        
        return RoutingResponse(
            provider=decision.provider.value,
            reason=decision.reason,
            estimated_cost=decision.estimated_cost,
            estimated_latency_ms=decision.estimated_latency_ms,
            fallback_provider=decision.fallback_provider.value if decision.fallback_provider else None,
            scores=decision.metadata.get("scores", {})
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Routing decision failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/estimate-cost", response_model=CostEstimateResponse)
async def estimate_cost(request: CostEstimateRequest):
    """Estimate cost for a job"""
    try:
        optimizer = get_cost_optimizer()
        
        provider = ProviderType(request.provider)
        job_type = JobType(request.job_type)
        
        estimate = optimizer.estimate_cost(
            provider=provider,
            job_type=job_type,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            gpu_type=request.gpu_type,
        )
        
        return CostEstimateResponse(
            provider=estimate.provider.value,
            estimated_cost_usd=estimate.estimated_cost_usd,
            estimated_duration_seconds=estimate.estimated_duration_seconds,
            confidence=estimate.confidence,
            breakdown=estimate.breakdown,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/budget")
async def get_budget_status():
    """Get current budget status"""
    optimizer = get_cost_optimizer()
    status = optimizer.get_budget_status()
    summary = optimizer.get_cost_summary()
    
    return {
        "daily": {
            "budget": status.daily_budget_usd,
            "spent": status.daily_spent_usd,
            "remaining": status.daily_remaining_usd,
        },
        "monthly": {
            "budget": status.monthly_budget_usd,
            "spent": status.monthly_spent_usd,
            "remaining": status.monthly_remaining_usd,
        },
        "is_over_budget": status.is_over_budget,
        "by_provider": summary,
    }


@router.get("/providers")
async def list_providers():
    """List available providers and their status"""
    try:
        provider_router = get_router()
        health = await provider_router.get_all_health()
        
        providers = []
        for provider_type in ProviderType:
            providers.append({
                "id": provider_type.value,
                "name": provider_type.name,
                "healthy": health.get(provider_type.value, False),
                "enabled": provider_type in provider_router.providers,
            })
        
        return {"providers": providers}
        
    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/providers/{provider_id}/health-check")
async def force_health_check(provider_id: str, background_tasks: BackgroundTasks):
    """Force a health check for a specific provider"""
    try:
        provider = ProviderType(provider_id)
        health_monitor = get_health_monitor()
        
        result = await health_monitor.force_check(provider)
        
        if result:
            return {
                "provider": provider.value,
                "status": result.status.value,
                "latency_ms": result.latency_ms,
                "error": result.error,
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Provider {provider_id} not registered"
            )
            
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {provider_id}")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """Get orchestrator metrics for monitoring"""
    optimizer = get_cost_optimizer()
    health_monitor = get_health_monitor()
    
    return {
        "cost": optimizer.get_cost_summary(),
        "health": health_monitor.get_health_summary(),
        "budget": optimizer.get_budget_status().__dict__,
    }
