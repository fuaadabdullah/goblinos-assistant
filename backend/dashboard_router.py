"""Dashboard API - Optimized endpoints for frontend dashboard with aggressive caching"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import os
import socket
import pathlib
import urllib.parse
import asyncio

from .database import get_db
from .models.provider import ProviderMetric
from .models.provider import Provider as RoutingProvider

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


# Simple in-memory cache with TTL
class SimpleCache:
    """Thread-safe in-memory cache with TTL"""

    def __init__(self):
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if datetime.now() < expiry:
                    return value
                else:
                    del self._cache[key]
            return None

    async def set(self, key: str, value: Any, ttl_seconds: int):
        """Set cached value with TTL"""
        async with self._lock:
            expiry = datetime.now() + timedelta(seconds=ttl_seconds)
            self._cache[key] = (value, expiry)

    async def clear(self):
        """Clear all cached entries"""
        async with self._lock:
            self._cache.clear()


# Global cache instance
cache = SimpleCache()


def cached(ttl_seconds: int):
    """Decorator for caching async endpoint results"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and args
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl_seconds)
            return result

        return wrapper

    return decorator


# Response Models
class ServiceStatus(BaseModel):
    """Individual service status."""

    status: str  # "healthy" | "degraded" | "down" | "unknown"
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    updated: str
    details: Optional[Dict[str, Any]] = None


class DashboardStatusResponse(BaseModel):
    """Compact dashboard status for all services"""

    backend_api: ServiceStatus
    vector_db: ServiceStatus
    mcp_servers: ServiceStatus
    rag_indexer: ServiceStatus
    sandbox_runner: ServiceStatus
    timestamp: str


class CostSummaryResponse(BaseModel):
    """Cached cost tracking summary"""

    total_cost: float
    cost_today: float
    cost_this_month: float
    by_provider: Dict[str, float]
    timestamp: str


# Helper Functions
def check_tcp_connection(
    host: str, port: int, timeout: float = 2.0
) -> tuple[bool, Optional[float]]:
    """Check TCP connection and measure latency"""
    try:
        import time

        start = time.time()
        s = socket.socket()
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        latency = (time.time() - start) * 1000  # Convert to ms
        return True, latency
    except Exception as e:
        return False, None


async def check_backend_status() -> ServiceStatus:
    """Check backend API health"""
    try:
        import httpx
        import time

        start = time.time()
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get("http://localhost:8001/health")
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                return ServiceStatus(
                    status="healthy",
                    latency_ms=round(latency, 2),
                    updated=datetime.now().isoformat(),
                )
            else:
                return ServiceStatus(
                    status="degraded",
                    latency_ms=round(latency, 2),
                    error=f"HTTP {response.status_code}",
                    updated=datetime.now().isoformat(),
                )
    except Exception as e:
        return ServiceStatus(
            status="down", error=str(e), updated=datetime.now().isoformat()
        )


async def check_vector_db_status() -> ServiceStatus:
    """Check vector database status - Supabase pgvector (primary) or Chroma (fallback)"""
    try:
        import time

        # Primary: Check Supabase pgvector
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv(
            "SUPABASE_ANON_KEY"
        )

        if supabase_url and supabase_key:
            try:
                import httpx

                start = time.time()
                # Check if rag_embeddings table exists via RPC health check
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Query the rag_embeddings table count
                    response = await client.get(
                        f"{supabase_url}/rest/v1/rag_embeddings?select=count",
                        headers={
                            "apikey": supabase_key,
                            "Authorization": f"Bearer {supabase_key}",
                            "Prefer": "count=exact",
                        },
                    )
                    latency = (time.time() - start) * 1000

                    if response.status_code == 200:
                        # Get count from header
                        count_header = response.headers.get("content-range", "0")
                        try:
                            total_docs = (
                                int(count_header.split("/")[-1])
                                if "/" in count_header
                                else 0
                            )
                        except (ValueError, IndexError):
                            total_docs = 0

                        return ServiceStatus(
                            status="healthy",
                            latency_ms=round(latency, 2),
                            updated=datetime.now().isoformat(),
                            details={
                                "backend": "supabase_pgvector",
                                "documents": total_docs,
                            },
                        )
                    elif response.status_code == 404:
                        # Table doesn't exist yet - need to run migration
                        return ServiceStatus(
                            status="degraded",
                            latency_ms=round(latency, 2),
                            updated=datetime.now().isoformat(),
                            details={
                                "backend": "supabase_pgvector",
                                "error": "embeddings table not found - run pgvector migration",
                            },
                        )
                    else:
                        return ServiceStatus(
                            status="down",
                            error=f"Supabase API error: {response.status_code}",
                            updated=datetime.now().isoformat(),
                        )
            except Exception as e:
                # Supabase check failed, fall through to Chroma check
                pass

        # Fallback: Check local Chroma
        chroma_path = os.getenv("CHROMA_DB_PATH")
        if not chroma_path:
            chroma_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "..",
                "data",
                "vector",
                "chroma",
                "chroma.sqlite3",
            )

        chroma_file = pathlib.Path(chroma_path).resolve()

        if chroma_file.exists():
            try:
                import chromadb

                start = time.time()
                client = chromadb.PersistentClient(path=str(chroma_file.parent))
                collections = client.list_collections()
                latency = (time.time() - start) * 1000

                total_docs = sum(
                    [col.count() for col in collections if hasattr(col, "count")]
                )

                return ServiceStatus(
                    status="healthy",
                    latency_ms=round(latency, 2),
                    updated=datetime.now().isoformat(),
                    details={
                        "backend": "chroma",
                        "collections": len(collections),
                        "documents": total_docs,
                    },
                )
            except ImportError:
                return ServiceStatus(
                    status="healthy",
                    updated=datetime.now().isoformat(),
                    details={"backend": "chroma", "collections": 0, "documents": 0},
                )

        # Check for hosted vector DB (Qdrant)
        qdrant_url = os.getenv("QDRANT_URL") or os.getenv("CHROMA_API_URL")
        if qdrant_url:
            parsed = urllib.parse.urlparse(qdrant_url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)

            connected, latency = check_tcp_connection(host, port)
            if connected:
                return ServiceStatus(
                    status="healthy",
                    latency_ms=latency,
                    updated=datetime.now().isoformat(),
                    details={"backend": "qdrant"},
                )
            else:
                return ServiceStatus(
                    status="down",
                    error="Connection refused",
                    updated=datetime.now().isoformat(),
                )

        # No vector DB configured
        return ServiceStatus(
            status="down",
            error="No vector DB configured (set SUPABASE_URL or CHROMA_DB_PATH)",
            updated=datetime.now().isoformat(),
        )
    except Exception as e:
        return ServiceStatus(
            status="down", error=str(e), updated=datetime.now().isoformat()
        )


async def check_mcp_status() -> ServiceStatus:
    """Check MCP server status"""
    try:
        active_servers = []

        # Check for configured MCP server URL (cloud deployment)
        mcp_url = os.getenv("MCP_SERVER_URL")
        if mcp_url:
            parsed = urllib.parse.urlparse(mcp_url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)

            connected, latency = check_tcp_connection(host, port)
            if connected:
                return ServiceStatus(
                    status="healthy",
                    latency_ms=latency,
                    updated=datetime.now().isoformat(),
                    details={"servers": [mcp_url], "count": 1, "mode": "cloud"},
                )

        # Check common MCP ports (local development)
        mcp_ports = [8765, 8766]
        for port in mcp_ports:
            connected, latency = check_tcp_connection("localhost", port, timeout=1.0)
            if connected:
                active_servers.append(f"localhost:{port}")

        if active_servers:
            return ServiceStatus(
                status="healthy",
                updated=datetime.now().isoformat(),
                details={
                    "servers": active_servers,
                    "count": len(active_servers),
                    "mode": "local",
                },
            )

        # In cloud deployment without MCP_SERVER_URL, MCP is optional
        if (
            os.getenv("FLY_APP_NAME")
            or os.getenv("RAILWAY_ENVIRONMENT")
            or os.getenv("RENDER")
        ):
            return ServiceStatus(
                status="degraded",
                updated=datetime.now().isoformat(),
                details={
                    "mode": "cloud",
                    "note": "MCP servers are optional in cloud deployment. Set MCP_SERVER_URL to enable.",
                },
            )

        return ServiceStatus(
            status="down",
            error="No MCP servers responding",
            updated=datetime.now().isoformat(),
        )
    except Exception as e:
        return ServiceStatus(
            status="down", error=str(e), updated=datetime.now().isoformat()
        )


async def check_rag_status() -> ServiceStatus:
    """Check RAG indexer status (EnhancedRAGService or TF-IDF fallback)"""
    try:
        import time

        start = time.time()

        # Check if core RAG dependencies are available
        rag_mode = None
        rag_available = False

        # Check for sentence-transformers (full RAG mode)
        try:
            from sentence_transformers import SentenceTransformer

            rag_mode = "sentence_transformers"
            rag_available = True
        except ImportError:
            pass

        # Check for TF-IDF fallback (sklearn)
        if not rag_available:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                rag_mode = "tfidf_fallback"
                rag_available = True
            except ImportError:
                pass

        latency = (time.time() - start) * 1000

        if rag_available:
            return ServiceStatus(
                status="healthy",
                latency_ms=round(latency, 2),
                updated=datetime.now().isoformat(),
                details={
                    "backend": "enhanced_rag",
                    "mode": rag_mode,
                },
            )

        # Fallback: Check for Raptor module (local development only)
        try:
            import sys
            from pathlib import Path

            sys.path.insert(
                0, str(Path(__file__).parent.parent.parent.parent / "GoblinOS")
            )
            from raptor_mini import raptor

            running = bool(raptor.running) if hasattr(raptor, "running") else False

            if running:
                return ServiceStatus(
                    status="healthy",
                    updated=datetime.now().isoformat(),
                    details={"backend": "raptor", "running": True},
                )
        except ImportError:
            pass

        # In cloud deployment, RAG is optional
        if (
            os.getenv("FLY_APP_NAME")
            or os.getenv("RAILWAY_ENVIRONMENT")
            or os.getenv("RENDER")
        ):
            return ServiceStatus(
                status="degraded",
                updated=datetime.now().isoformat(),
                details={
                    "note": "RAG indexer is optional in cloud deployment. Use /api/rag endpoints for basic retrieval.",
                },
            )

        return ServiceStatus(
            status="down",
            error="RAG service not available",
            updated=datetime.now().isoformat(),
        )
    except Exception as e:
        return ServiceStatus(
            status="down", error=str(e), updated=datetime.now().isoformat()
        )


async def check_sandbox_status() -> ServiceStatus:
    """Check sandbox runner status"""
    try:
        active_jobs = 0
        queue_size = 0

        # Try Redis-backed task queue
        try:
            import redis

            REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            r = redis.from_url(REDIS_URL)
            keys = r.keys("task:*")

            for key in keys:
                k = key.decode("utf-8")
                if ":logs" in k or ":artifacts" in k:
                    continue
                # Simple count - could be enhanced with actual status checks
                active_jobs += 1
        except Exception:
            # Fall back to in-memory TASKS
            try:
                from execute_router import TASKS

                for job_id, info in TASKS.items():
                    if info.get("status") == "running":
                        active_jobs += 1
                    elif info.get("status") == "queued":
                        queue_size += 1
            except Exception:
                # Fallback to 0 if TASKS not available
                active_jobs = 0
                queue_size = 0

        status = "degraded" if queue_size > 5 else "healthy"

        return ServiceStatus(
            status=status,
            updated=datetime.now().isoformat(),
            details={"active_jobs": active_jobs, "queue_size": queue_size},
        )
    except Exception as e:
        return ServiceStatus(
            status="down", error=str(e), updated=datetime.now().isoformat()
        )


# Main Endpoints
@router.get("/status", response_model=DashboardStatusResponse)
@cached(ttl_seconds=10)  # Cache for 10 seconds - frequent updates for status
async def get_dashboard_status():
    """
    Get compact status for all services - optimized for dashboard display.

    Cached for 10 seconds to balance freshness with performance.
    Includes latency measurements and error details.
    """
    try:
        # Gather all service statuses in parallel
        import asyncio

        backend, vector_db, mcp, rag, sandbox = await asyncio.gather(
            check_backend_status(),
            check_vector_db_status(),
            check_mcp_status(),
            check_rag_status(),
            check_sandbox_status(),
            return_exceptions=True,
        )

        # Handle any exceptions from parallel checks
        if isinstance(backend, Exception):
            backend = ServiceStatus(
                status="unknown", error=str(backend), updated=datetime.now().isoformat()
            )
        if isinstance(vector_db, Exception):
            vector_db = ServiceStatus(
                status="unknown",
                error=str(vector_db),
                updated=datetime.now().isoformat(),
            )
        if isinstance(mcp, Exception):
            mcp = ServiceStatus(
                status="unknown", error=str(mcp), updated=datetime.now().isoformat()
            )
        if isinstance(rag, Exception):
            rag = ServiceStatus(
                status="unknown", error=str(rag), updated=datetime.now().isoformat()
            )
        if isinstance(sandbox, Exception):
            sandbox = ServiceStatus(
                status="unknown", error=str(sandbox), updated=datetime.now().isoformat()
            )

        return DashboardStatusResponse(
            backend_api=backend,
            vector_db=vector_db,
            mcp_servers=mcp,
            rag_indexer=rag,
            sandbox_runner=sandbox,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get dashboard status: {str(e)}"
        )


@router.get("/costs", response_model=CostSummaryResponse)
@cached(ttl_seconds=60)  # Cache for 60 seconds - costs don't change rapidly
async def get_cost_summary(db: Session = Depends(get_db)):
    """
    Get aggregated cost tracking summary.

    Cached for 60 seconds - cost data changes slowly, aggressive caching reduces DB load.
    """
    try:
        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day)
        month_start = datetime(now.year, now.month, 1)

        # Query total costs
        total_cost = float(
            db.query(func.sum(ProviderMetric.cost_incurred))
            .filter(ProviderMetric.cost_incurred.isnot(None))
            .scalar()
            or 0.0
        )

        # Today's costs
        cost_today = float(
            db.query(func.sum(ProviderMetric.cost_incurred))
            .filter(
                ProviderMetric.cost_incurred.isnot(None),
                ProviderMetric.timestamp >= today_start,
            )
            .scalar()
            or 0.0
        )

        # This month's costs
        cost_this_month = float(
            db.query(func.sum(ProviderMetric.cost_incurred))
            .filter(
                ProviderMetric.cost_incurred.isnot(None),
                ProviderMetric.timestamp >= month_start,
            )
            .scalar()
            or 0.0
        )

        # Cost by provider
        by_provider = {}
        providers = db.query(RoutingProvider).all()
        for provider in providers:
            provider_cost = float(
                db.query(func.sum(ProviderMetric.cost_incurred))
                .filter(
                    ProviderMetric.provider_id == provider.id,
                    ProviderMetric.cost_incurred.isnot(None),
                )
                .scalar()
                or 0.0
            )
            if provider_cost > 0:  # Only include providers with actual costs
                by_provider[provider.display_name] = provider_cost

        return CostSummaryResponse(
            total_cost=round(total_cost, 4),
            cost_today=round(cost_today, 4),
            cost_this_month=round(cost_this_month, 4),
            by_provider=by_provider,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get cost summary: {str(e)}"
        )


@router.get("/metrics/{service}")
@cached(ttl_seconds=30)  # Cache for 30 seconds
async def get_service_metrics(service: str, db: Session = Depends(get_db)):
    """
    Get detailed metrics for a specific service.

    Cached for 30 seconds - balances freshness with performance.
    Returns latency history, error rates, and other service-specific metrics.
    """
    try:
        cutoff_time = datetime.now() - timedelta(hours=24)

        if service == "backend":
            # Get recent provider metrics as proxy for backend performance
            metrics = (
                db.query(ProviderMetric.timestamp, ProviderMetric.response_time_ms)
                .filter(
                    ProviderMetric.timestamp >= cutoff_time,
                    ProviderMetric.response_time_ms.isnot(None),
                )
                .order_by(ProviderMetric.timestamp.desc())
                .limit(100)
                .all()
            )

            timestamps = [m[0].isoformat() for m in metrics]
            latencies = [float(m[1]) for m in metrics]

            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            return {
                "service": service,
                "latency_history": {
                    "timestamps": timestamps,
                    "latencies": latencies,
                },
                "avg_latency_ms": round(avg_latency, 2),
                "data_points": len(latencies),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            # Placeholder for other services
            return {
                "service": service,
                "message": f"Detailed metrics not yet implemented for {service}",
                "timestamp": datetime.now().isoformat(),
            }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get service metrics: {str(e)}"
        )
