from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

try:
    from ..database import get_db
except ImportError:  # pragma: no cover - fallback for tests/import contexts
    from database import get_db

try:
    from ..config import settings
except ImportError:  # pragma: no cover - fallback for tests/import contexts
    from config import settings

# Import auth components for health checks
try:
    from ..auth.challenge_store import get_challenge_store_instance

    challenge_store_available = True
except ImportError:
    challenge_store_available = False

# Import session cache for health checks
try:
    from ..cache.session_cache import get_session_cache

    session_cache_available = True
except ImportError:
    session_cache_available = False

router = APIRouter()


class ComprehensiveHealthResponse(BaseModel):
    """Comprehensive health check response with environment awareness"""

    status: str  # 'healthy', 'degraded', 'unhealthy'
    timestamp: str
    environment: str
    components: dict[str, object]


@router.get("/", response_model=ComprehensiveHealthResponse)
async def comprehensive_health_check(  # noqa: C901
    db: Session = Depends(get_db),  # noqa: B008
):
    """
    Comprehensive health check with environment-aware status reporting.

    Checks critical components: Redis, database, auth routes, and environment safety.
    """
    checks: dict[str, object] = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment,
        "components": {},
    }

    # Redis/Challenge Store check
    if challenge_store_available:
        try:
            challenge_store = get_challenge_store_instance()
            redis_health = await challenge_store.health_check()

            # Redis-backed store returns `redis_available`; in-memory fallback returns `healthy`.
            redis_available = bool(
                redis_health.get("redis_available")
                if "redis_available" in redis_health
                else redis_health.get("healthy", False)
            )
            store_type = redis_health.get(
                "store_type",
                "redis" if "redis_available" in redis_health else "unknown",
            )

            # In production, in-memory challenges are acceptable only for single-instance deployments.
            # Mark as degraded to reflect reduced safety (no cross-instance consistency).
            if store_type == "in_memory" and settings.is_production:
                component_status = "degraded" if settings.instance_count <= 1 else "unhealthy"
            else:
                component_status = "healthy" if redis_available else "degraded"

            checks["components"]["redis"] = {
                "status": component_status,
                "store_type": store_type,
                "redis_available": redis_available,
            }

            if store_type == "in_memory" and settings.is_production and settings.instance_count > 1:
                checks["components"]["redis"]["alert"] = (
                    "CRITICAL: In-memory challenge store in multi-instance production"
                )

        except Exception as e:
            checks["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
            checks["status"] = "unhealthy"
    else:
        checks["components"]["redis"] = {
            "status": "unhealthy",
            "error": "Challenge store not available",
        }
        checks["status"] = "unhealthy"

    # Session Cache check
    if session_cache_available:
        try:
            session_cache = get_session_cache()
            cache_health = session_cache.health_check()

            checks["components"]["session_cache"] = {
                "status": "healthy" if cache_health["available"] else "degraded",
                "memory_used": cache_health.get("memory_used", "unknown"),
                "session_cache_ttl": cache_health.get("session_cache_ttl", "unknown"),
            }

        except Exception as e:
            checks["components"]["session_cache"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            checks["status"] = "unhealthy"
    else:
        checks["components"]["session_cache"] = {
            "status": "unhealthy",
            "error": "Session cache not available",
        }
        checks["status"] = "unhealthy"

    # Database check
    try:
        # Simple connectivity test
        result = db.execute(text("SELECT 1")).fetchone()
        if result:
            checks["components"]["database"] = {"status": "healthy"}
        else:
            checks["components"]["database"] = {
                "status": "degraded",
                "message": "Database query returned no results",
            }
    except Exception as e:
        checks["components"]["database"] = {"status": "unhealthy", "error": str(e)}
        checks["status"] = "unhealthy"

    # Auth routes check
    try:
        # Import the app via package path so it works in production.
        try:
            from ..main import app  # type: ignore
        except ImportError:  # pragma: no cover - fallback for tests/import contexts
            from main import app  # type: ignore

        auth_routes = [
            r
            for r in app.routes
            if hasattr(r, "path")
            and (
                str(r.path).startswith("/v1/auth")
                or str(r.path).startswith("/auth")  # legacy/non-versioned
            )
        ]
        checks["components"]["auth_routes"] = {
            "status": "healthy" if len(auth_routes) > 0 else "unhealthy",
            "count": len(auth_routes),
            "routes": [r.path for r in auth_routes[:5]],  # Show first 5 routes
        }
        if len(auth_routes) == 0:
            checks["status"] = "unhealthy"
    except Exception as e:
        checks["components"]["auth_routes"] = {
            "status": "unhealthy",
            "error": f"Failed to check routes: {str(e)}",
        }
        checks["status"] = "unhealthy"

    # Configuration validation
    config_issues = []
    if settings.is_production and not settings.database_url:
        config_issues.append("DATABASE_URL required in production")

    if settings.is_production and settings.allow_memory_fallback and settings.is_multi_instance:
        config_issues.append("Memory fallback not allowed in multi-instance production")

    checks["components"]["configuration"] = {
        "status": "healthy" if not config_issues else "unhealthy",
        "issues": config_issues,
        "environment": settings.environment,
        "multi_instance": settings.is_multi_instance,
    }

    if config_issues:
        checks["status"] = "unhealthy"

    # Overall status determination
    component_statuses = [
        c.get("status", "unknown")
        for c in checks["components"].values()  # type: ignore[call-arg]
    ]

    if "unhealthy" in component_statuses:
        checks["status"] = "unhealthy"
    elif "degraded" in component_statuses:
        checks["status"] = "degraded"
    # Otherwise remains 'healthy'

    return checks


@router.get("", response_model=ComprehensiveHealthResponse, include_in_schema=False)
async def comprehensive_health_check_no_trailing_slash(
    db: Session = Depends(get_db),  # noqa: B008
):
    """Alias for /health without redirect-to-slash (avoids https->http mixed redirects behind proxies)."""
    return await comprehensive_health_check(db)


@router.get("/health", response_model=ComprehensiveHealthResponse)
async def simple_health_check(
    db: Session = Depends(get_db),  # noqa: B008
):
    """
    Simple health check endpoint for load balancers and monitoring systems.
    Same as comprehensive check but accessible at /health path.
    """
    return await comprehensive_health_check(db)


# ---------------------------------------------------------------------------
# Frontend Compatibility Endpoints
# ---------------------------------------------------------------------------


@router.get("/streaming")
async def streaming_health():
    """Compatibility endpoint expected by the frontend."""
    return {
        "status": "healthy",
        "service": "streaming",
        "endpoint": "/stream",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/routing")
async def routing_health(
    db: Session = Depends(get_db),  # noqa: B008
):
    """Compatibility endpoint expected by the frontend."""
    try:
        try:
            from ..models.provider import Provider as RoutingProvider
        except ImportError:  # pragma: no cover - fallback for tests/import contexts
            from models.provider import Provider as RoutingProvider

        total = db.query(RoutingProvider).count()
        enabled = db.query(RoutingProvider).filter(RoutingProvider.enabled.is_(True)).count()
        return {
            "status": "healthy" if enabled > 0 else "degraded",
            "service": "routing",
            "providers": {"total": total, "enabled": enabled},
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "service": "routing",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        }


__all__ = ["router", "ComprehensiveHealthResponse"]
