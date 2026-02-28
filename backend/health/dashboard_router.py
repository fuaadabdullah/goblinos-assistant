from __future__ import annotations

import os
import pathlib
import socket
from datetime import datetime, timedelta

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.provider import Provider as RoutingProvider
from ..models.provider import ProviderMetric

try:
    import celery_task_queue  # noqa: I001
    import redis  # noqa: I001
except Exception:
    celery_task_queue = None
    redis = None

router = APIRouter()
db_dependency = Depends(get_db)


class ChromaStatusResponse(BaseModel):
    status: str
    collections: int
    documents: int
    last_check: str


class MCPStatusResponse(BaseModel):
    status: str
    servers: list[str]
    active_connections: int
    last_check: str


class RaptorStatusResponse(BaseModel):
    status: str
    running: bool
    config_file: str
    last_check: str


class SandboxStatusResponse(BaseModel):
    status: str
    active_jobs: int
    queue_size: int
    last_check: str


class SchedulerStatusResponse(BaseModel):
    status: str
    jobs: list[dict[str, object]]
    last_check: str


class CostTrackingResponse(BaseModel):
    total_cost: float
    cost_today: float
    cost_this_month: float
    by_provider: dict[str, float]


class LatencyHistoryResponse(BaseModel):
    timestamps: list[str]
    latencies: list[float]


class ServiceError(BaseModel):
    timestamp: str
    message: str
    service: str


class RetestServiceResponse(BaseModel):
    success: bool
    latency: float | None
    message: str


@router.get("/chroma/status", response_model=ChromaStatusResponse)
async def get_chroma_status() -> ChromaStatusResponse:
    """Get detailed Chroma vector database status."""
    try:
        chroma_path = os.getenv("CHROMA_DB_PATH")
        if not chroma_path:
            chroma_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "data",
                "vector",
                "chroma",
                "chroma.sqlite3",
            )

        chroma_file = pathlib.Path(chroma_path).resolve()

        if chroma_file.exists():
            # Try to import chromadb and get actual collection stats
            try:
                import chromadb

                client = chromadb.PersistentClient(path=str(chroma_file.parent))
                collections = client.list_collections()
                total_docs = sum([col.count() for col in collections if hasattr(col, "count")])

                return ChromaStatusResponse(
                    status="healthy",
                    collections=len(collections),
                    documents=total_docs,
                    last_check=datetime.now().isoformat(),
                )
            except ImportError:
                # Fallback if chromadb not available - just check file exists
                return ChromaStatusResponse(
                    status="healthy",
                    collections=0,
                    documents=0,
                    last_check=datetime.now().isoformat(),
                )
        return ChromaStatusResponse(
            status="down",
            collections=0,
            documents=0,
            last_check=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Chroma status: {str(e)}") from e


@router.get("/mcp/status", response_model=MCPStatusResponse)
async def get_mcp_status() -> MCPStatusResponse:
    """Get MCP (Model Context Protocol) server status."""
    try:
        # Check for MCP server configuration
        mcp_servers: list[str] = []
        active_connections = 0

        if os.getenv("MCP_SERVER_URL"):
            mcp_servers.append("primary")
            active_connections += 1

        # Check if local MCP servers are running
        mcp_ports = [8765, 8766]  # Common MCP ports
        for port in mcp_ports:
            try:
                sock = socket.socket()
                sock.settimeout(1)
                sock.connect(("localhost", port))
                sock.close()
                mcp_servers.append(f"localhost:{port}")
                active_connections += 1
            except Exception:
                pass

        status = "healthy" if active_connections > 0 else "down"

        return MCPStatusResponse(
            status=status,
            servers=mcp_servers,
            active_connections=active_connections,
            last_check=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get MCP status: {str(e)}") from e


@router.get("/raptor/status", response_model=RaptorStatusResponse)
async def get_raptor_status() -> RaptorStatusResponse:
    """Get RAG indexer (Raptor) status."""
    try:
        import sys
        from pathlib import Path

        # Add GoblinOS to path for raptor import
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "GoblinOS"))
        from raptor_mini import raptor

        running = bool(raptor.running) if hasattr(raptor, "running") else False
        config_file = getattr(raptor, "ini_path", "config/raptor.ini")

        status = "healthy" if running else "down"

        return RaptorStatusResponse(
            status=status,
            running=running,
            config_file=config_file,
            last_check=datetime.now().isoformat(),
        )
    except Exception:
        # If raptor can't be imported or checked, return down status
        return RaptorStatusResponse(
            status="down",
            running=False,
            config_file="unknown",
            last_check=datetime.now().isoformat(),
        )


def _count_redis_tasks() -> tuple[int, int] | None:
    if celery_task_queue is None or redis is None:
        return None

    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.from_url(redis_url)
        keys = client.keys("task:*")
    except Exception:
        return None

    active_jobs = 0
    queue_size = 0
    for key in keys:
        k = key.decode("utf-8")
        if ":logs" in k or ":artifacts" in k:
            continue
        task_id = k.split(":", 1)[1]
        meta = celery_task_queue.get_task_meta(task_id)
        status = meta.get("status")
        if status == "running":
            active_jobs += 1
        elif status == "queued":
            queue_size += 1
    return active_jobs, queue_size


def _count_in_memory_tasks() -> tuple[int, int]:
    try:
        from ..execute_router import TASKS
    except Exception:
        return 0, 0

    active_jobs = 0
    queue_size = 0
    for _job_id, info in TASKS.items():
        if info.get("status") == "running":
            active_jobs += 1
        elif info.get("status") == "queued":
            queue_size += 1
    return active_jobs, queue_size


@router.get("/sandbox/status", response_model=SandboxStatusResponse)
async def get_sandbox_status() -> SandboxStatusResponse:
    """Get sandbox runner status."""
    try:
        counts = _count_redis_tasks()
        if counts is None:
            active_jobs, queue_size = _count_in_memory_tasks()
        else:
            active_jobs, queue_size = counts

        status = "healthy" if active_jobs >= 0 else "down"

        return SandboxStatusResponse(
            status=status,
            active_jobs=active_jobs,
            queue_size=queue_size,
            last_check=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get sandbox status: {str(e)}"
        ) from e


@router.get("/scheduler/status", response_model=SchedulerStatusResponse)
async def get_scheduler_status() -> SchedulerStatusResponse:
    """Get APScheduler status and job information."""
    try:
        from ..scheduler import get_scheduler_status

        scheduler_info = get_scheduler_status()

        return SchedulerStatusResponse(
            status=scheduler_info.get("status", "unknown"),
            jobs=scheduler_info.get("jobs", []),
            last_check=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get scheduler status: {str(e)}"
        ) from e


@router.get("/cost-tracking", response_model=CostTrackingResponse)
async def get_cost_tracking(db: Session = db_dependency) -> CostTrackingResponse:
    """Get aggregated cost tracking across providers."""
    try:
        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day)
        month_start = datetime(now.year, now.month, 1)

        total_cost_query = (
            db.query(func.sum(ProviderMetric.cost_incurred))
            .filter(ProviderMetric.cost_incurred.isnot(None))
            .scalar()
        )
        total_cost = float(total_cost_query or 0.0)

        cost_today_query = (
            db.query(func.sum(ProviderMetric.cost_incurred))
            .filter(
                ProviderMetric.cost_incurred.isnot(None),
                ProviderMetric.timestamp >= today_start,
            )
            .scalar()
        )
        cost_today = float(cost_today_query or 0.0)

        cost_month_query = (
            db.query(func.sum(ProviderMetric.cost_incurred))
            .filter(
                ProviderMetric.cost_incurred.isnot(None),
                ProviderMetric.timestamp >= month_start,
            )
            .scalar()
        )
        cost_this_month = float(cost_month_query or 0.0)

        by_provider: dict[str, float] = {}
        providers = db.query(RoutingProvider).all()
        for provider in providers:
            provider_cost = (
                db.query(func.sum(ProviderMetric.cost_incurred))
                .filter(
                    ProviderMetric.provider_id == provider.id,
                    ProviderMetric.cost_incurred.isnot(None),
                )
                .scalar()
            )
            by_provider[provider.display_name] = float(provider_cost or 0.0)

        return CostTrackingResponse(
            total_cost=total_cost,
            cost_today=cost_today,
            cost_this_month=cost_this_month,
            by_provider=by_provider,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cost tracking: {str(e)}") from e


@router.get("/latency-history/{service}", response_model=LatencyHistoryResponse)
async def get_latency_history(
    service: str, hours: int = 24, db: Session = db_dependency
) -> LatencyHistoryResponse:
    """Get latency history for a service over the specified hours."""
    try:
        cutoff_time = datetime.now() - timedelta(hours=hours)

        if service == "backend":
            metrics = (
                db.query(ProviderMetric.timestamp, ProviderMetric.response_time_ms)
                .filter(
                    ProviderMetric.timestamp >= cutoff_time,
                    ProviderMetric.response_time_ms.isnot(None),
                )
                .order_by(ProviderMetric.timestamp)
                .limit(100)
                .all()
            )

            timestamps = [m[0].isoformat() for m in metrics]
            latencies = [float(m[1]) for m in metrics]
        elif service == "chroma":
            timestamps = []
            latencies = []
        else:
            timestamps = []
            latencies = []

        return LatencyHistoryResponse(timestamps=timestamps, latencies=latencies)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get latency history: {str(e)}"
        ) from e


@router.get("/service-errors/{service}", response_model=list[ServiceError])
async def get_service_errors(service: str, limit: int = 10) -> list[ServiceError]:
    """Get recent errors for a specific service."""
    try:
        errors: list[ServiceError] = []

        if service == "backend":
            log_file = os.path.join(os.path.dirname(__file__), "logs", "app.log")
        elif service == "chroma":
            log_file = os.path.join(os.path.dirname(__file__), "logs", "chroma.log")
        elif service == "raptor":
            log_file = os.path.join(os.path.dirname(__file__), "logs", "raptor.log")
        else:
            log_file = None

        if log_file and os.path.exists(log_file):
            with open(log_file) as f:
                lines = f.readlines()
                error_lines = [line for line in lines if "error" in line.lower()][-limit:]

                for line in error_lines:
                    errors.append(
                        ServiceError(
                            timestamp=datetime.now().isoformat(),
                            message=line.strip(),
                            service=service,
                        )
                    )

        return errors
    except Exception:
        return []


async def _retest_backend() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            return response.status_code == 200
    except Exception:
        return False


async def _retest_status(endpoint_call) -> bool:
    try:
        response = await endpoint_call()
        return response.status == "healthy"
    except Exception:
        return False


@router.post("/retest/{service}", response_model=RetestServiceResponse)
async def retest_service(service: str) -> RetestServiceResponse:
    """Trigger a health retest for a specific service."""
    try:
        handlers = {
            "backend": _retest_backend,
            "chroma": lambda: _retest_status(get_chroma_status),
            "mcp": lambda: _retest_status(get_mcp_status),
            "raptor": lambda: _retest_status(get_raptor_status),
            "sandbox": lambda: _retest_status(get_sandbox_status),
        }

        handler = handlers.get(service)
        if handler is None:
            raise HTTPException(status_code=400, detail=f"Unknown service: {service}")

        start_time = datetime.now()
        success = await handler()
        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds() * 1000

        message = f"{service.capitalize()} is {'healthy' if success else 'unhealthy'}"

        return RetestServiceResponse(success=success, latency=latency, message=message)
    except Exception as e:
        return RetestServiceResponse(
            success=False, latency=None, message=f"Retest failed: {str(e)}"
        )


__all__ = [
    "router",
    "ChromaStatusResponse",
    "MCPStatusResponse",
    "RaptorStatusResponse",
    "SandboxStatusResponse",
    "SchedulerStatusResponse",
    "CostTrackingResponse",
    "LatencyHistoryResponse",
    "ServiceError",
    "RetestServiceResponse",
]
