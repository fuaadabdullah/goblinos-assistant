import os
import pathlib
import socket
import urllib.parse

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

try:
    from ..database import get_db
except ImportError:  # pragma: no cover - fallback for tests/import contexts
    from database import get_db

# Import LLM adapters for health checks
try:
    from ..providers.anthropic_adapter import AnthropicAdapter
    from ..providers.deepseek_adapter import DeepSeekAdapter
    from ..providers.gemini_adapter import GeminiAdapter
    from ..providers.grok_adapter import GrokAdapter
    from ..providers.ollama_adapter import OllamaAdapter
    from ..providers.openai_adapter import OpenAIAdapter
except ImportError:
    # Adapters may not be available in all environments
    OllamaAdapter = None
    OpenAIAdapter = None
    AnthropicAdapter = None
    GrokAdapter = None
    DeepSeekAdapter = None
    GeminiAdapter = None

router = APIRouter()


class HealthCheckResponse(BaseModel):
    status: str
    checks: dict[str, object]


async def check_database_health(db_url: str) -> dict[str, object]:
    """Perform comprehensive database health check."""
    parsed = urllib.parse.urlparse(db_url)
    scheme = (parsed.scheme or "").lower()

    if not scheme.startswith("postgres"):
        return {"status": "unhealthy", "error": "Unsupported database type"}

    host = parsed.hostname
    port = parsed.port or 5432
    if not host:
        return {"status": "unhealthy", "error": "Database URL is missing host"}

    # TCP connectivity check (supports IPv4/IPv6)
    try:
        socket.create_connection((host, port), timeout=5).close()
    except TimeoutError:
        return {"status": "unhealthy", "error": "Database connection timeout"}
    except socket.gaierror:
        return {"status": "unhealthy", "error": "Database host resolution failed"}
    except Exception as e:
        return {
            "status": "unhealthy",
            "type": "postgresql",
            "host": host,
            "port": port,
            "error": f"Database TCP connectivity failed: {str(e)}",
        }

    # Test actual database connection and query via SQLAlchemy Session.
    db = next(get_db())
    try:
        result = db.execute("SELECT 1 as health_check").fetchone()
        if result and result[0] == 1:
            return {
                "status": "healthy",
                "type": "postgresql",
                "host": host,
                "port": port,
                "connection_test": "passed",
                "query_test": "passed",
            }
        return {
            "status": "degraded",
            "type": "postgresql",
            "host": host,
            "port": port,
            "connection_test": "passed",
            "query_test": "failed",
        }
    except Exception as e:
        return {
            "status": "degraded",
            "type": "postgresql",
            "host": host,
            "port": port,
            "connection_test": "passed",
            "query_test": "failed",
            "error": str(e),
        }
    finally:
        db.close()


async def check_llm_provider_health(  # noqa: C901
    provider_name: str, api_key: str, base_url: str
) -> dict[str, object]:
    """Perform comprehensive health check for LLM provider."""
    try:
        normalized_name = provider_name.lower().replace("-", "_")

        if normalized_name == "ollama" and OllamaAdapter:
            # Test Ollama connectivity with detailed checks
            adapter = OllamaAdapter(api_key, base_url)
            models = await adapter.list_models()

            if models and len(models) > 0:
                # Test a simple inference call to verify functionality
                test_model = models[0].get("id", "")
                if test_model:
                    try:
                        # Quick test with minimal tokens to verify inference works
                        test_response = await adapter.chat(
                            model=test_model,
                            messages=[{"role": "user", "content": "Hello"}],
                            max_tokens=5,
                            temperature=0.0,
                        )
                        if test_response and len(test_response.strip()) > 0:
                            return {
                                "status": "healthy",
                                "models_available": len(models),
                                "sample_models": [m.get("id", "") for m in models[:3]],
                                "inference_test": "passed",
                                "response_time_ms": None,  # Could add timing here
                            }
                        return {
                            "status": "degraded",
                            "models_available": len(models),
                            "inference_test": "failed",
                            "error": "Empty response from inference test",
                        }
                    except Exception as e:
                        return {
                            "status": "degraded",
                            "models_available": len(models),
                            "inference_test": "failed",
                            "error": f"Inference test failed: {str(e)}",
                        }
                return {
                    "status": "degraded",
                    "models_available": len(models),
                    "inference_test": "skipped",
                    "error": "No valid model ID found",
                }
            return {"status": "unhealthy", "error": "No models available"}

        if normalized_name in {
            "openai",
            "aliyun",
            "alibaba",
            "alibaba_cloud",
            "azure",
            "azure_openai",
        } and OpenAIAdapter:
            # Test OpenAI connectivity with models list and basic inference
            adapter = OpenAIAdapter(api_key, base_url)
            models = await adapter.list_models()

            if models and len(models) > 0:
                # Test inference with an available chat model
                model_ids = {m.get("id") for m in models if isinstance(m, dict)}
                preferred = [
                    "gpt-4o-mini",
                    "gpt-4o",
                    "gpt-3.5-turbo",
                    "gpt-4-turbo",
                    "gpt-4",
                ]
                test_model = next(
                    (m for m in preferred if m in model_ids),
                    models[0].get("id") if isinstance(models[0], dict) else None,
                )
                try:
                    test_response = await adapter.chat(
                        model=test_model or "gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5,
                        temperature=0.0,
                    )
                    if test_response and test_response.strip():
                        return {
                            "status": "healthy",
                            "models_available": len(models),
                            "inference_test": "passed",
                            "test_model": test_model,
                        }
                    return {
                        "status": "degraded",
                        "models_available": len(models),
                        "inference_test": "failed",
                        "test_model": test_model,
                        "error": "Empty response from inference test",
                    }
                except Exception as e:
                    return {
                        "status": "degraded",
                        "models_available": len(models),
                        "inference_test": "failed",
                        "error": str(e),
                    }
            return {"status": "unhealthy", "error": "No models available"}

        if normalized_name == "anthropic" and AnthropicAdapter:
            # Test Anthropic connectivity
            adapter = AnthropicAdapter(api_key, base_url)
            try:
                # Test with Claude model
                test_response = await adapter.chat(
                    model="claude-3-haiku-20240307",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                    temperature=0.0,
                )
                if test_response and test_response.strip():
                    return {"status": "healthy", "inference_test": "passed"}
                return {
                    "status": "degraded",
                    "inference_test": "failed",
                    "error": "Empty response from inference test",
                }
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}

        if normalized_name == "deepseek" and DeepSeekAdapter:
            adapter = DeepSeekAdapter(api_key, base_url)
            models = await adapter.list_models()
            test_model = "deepseek-chat"
            if models and isinstance(models[0], dict):
                test_model = models[0].get("id") or test_model
            try:
                test_response = await adapter.chat(
                    model=test_model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                    temperature=0.0,
                )
                if test_response and test_response.strip():
                    return {
                        "status": "healthy",
                        "models_available": len(models) if models else 0,
                        "inference_test": "passed",
                        "test_model": test_model,
                    }
                return {
                    "status": "degraded",
                    "models_available": len(models) if models else 0,
                    "inference_test": "failed",
                    "test_model": test_model,
                    "error": "Empty response from inference test",
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "models_available": len(models) if models else 0,
                    "inference_test": "failed",
                    "error": str(e),
                }

        if normalized_name == "gemini" and GeminiAdapter:
            adapter = GeminiAdapter(api_key, base_url)
            try:
                test = await adapter.test_completion(model="gemini-pro", max_tokens=10)
                if test.get("success"):
                    return {"status": "healthy", "inference_test": "passed"}
                return {
                    "status": "degraded",
                    "inference_test": "failed",
                    "error": test.get("error", "Inference test failed"),
                }
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}

        # Fallback to basic HTTP connectivity test
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{base_url}/health", timeout=5)
            if response.status_code < 400:
                return {"status": "healthy", "http_status": response.status_code}
            return {"status": "unhealthy", "http_status": response.status_code}

    except httpx.TimeoutException:
        return {"status": "unhealthy", "error": "Request timeout"}
    except httpx.ConnectError:
        return {"status": "unhealthy", "error": "Connection failed"}
    except Exception as e:
        return {"status": "unhealthy", "error": f"Health check failed: {str(e)}"}


@router.get("/all", response_model=HealthCheckResponse)
async def health_all():  # noqa: C901
    """Perform full health checks: database, vector DB, and providers"""
    checks: dict[str, object] = {}

    # DB check with comprehensive testing
    db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_URL")
    if db_url:
        checks["database"] = await check_database_health(db_url)
    else:
        checks["database"] = {
            "status": "skipped",
            "reason": "DATABASE_URL or SUPABASE_URL not set",
        }

    # Vector DB check (Supabase pgvector, local Chroma sqlite, or hosted vector DB)
    try:
        # Prefer Supabase/pgvector when SUPABASE_URL is configured.
        supabase_url = os.getenv("SUPABASE_URL")
        if supabase_url:
            try:
                db = next(get_db())
                try:
                    exists = bool(
                        db.execute("SELECT to_regclass('public.embeddings') IS NOT NULL").scalar()
                    )
                    if not exists:
                        checks["vector_db"] = {
                            "status": "degraded",
                            "backend": "supabase_pgvector",
                            "error": "embeddings table not found - run pgvector migration",
                        }
                    else:
                        # Lightweight accessibility check (avoid COUNT(*) table scans)
                        db.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone()
                        checks["vector_db"] = {
                            "status": "healthy",
                            "backend": "supabase_pgvector",
                        }
                finally:
                    db.close()
            except Exception as e:
                checks["vector_db"] = {
                    "status": "unhealthy",
                    "backend": "supabase_pgvector",
                    "error": str(e),
                }
        else:
            # Fallback: Chroma sqlite file (local) or hosted vector DB (Qdrant/Chroma cloud)
            chroma_path = os.getenv("CHROMA_DB_PATH")
            if not chroma_path:
                # Prefer the Fly volume mount when present.
                chroma_path = "/app/data/vector/chroma/chroma.sqlite3"
                if not pathlib.Path("/app/data").exists():
                    # default path in repo (local dev)
                    chroma_path = os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "data",
                        "vector",
                        "chroma",
                        "chroma.sqlite3",
                    )

            chroma_candidate = pathlib.Path(chroma_path).resolve()
            chroma_file = (
                (chroma_candidate / "chroma.sqlite3").resolve()
                if chroma_candidate.is_dir()
                else chroma_candidate
            )

            if chroma_file.exists():
                checks["vector_db"] = {
                    "status": "healthy",
                    "backend": "chroma",
                    "path": str(chroma_file),
                }
            else:
                qdrant_url = os.getenv("QDRANT_URL") or os.getenv("CHROMA_API_URL")
                if qdrant_url:
                    try:
                        parsed = urllib.parse.urlparse(qdrant_url)
                        host = parsed.hostname
                        port = parsed.port or (443 if parsed.scheme == "https" else 80)
                        socket.create_connection((host, port), timeout=3).close()
                        checks["vector_db"] = {
                            "status": "healthy",
                            "backend": "hosted",
                            "host": host,
                            "port": port,
                            "url": qdrant_url,
                        }
                    except Exception as e:
                        checks["vector_db"] = {
                            "status": "unhealthy",
                            "backend": "hosted",
                            "url": qdrant_url,
                            "error": str(e),
                        }
                else:
                    checks["vector_db"] = {
                        "status": "unhealthy",
                        "backend": "chroma",
                        "path": str(chroma_file),
                        "error": "file not found; set SUPABASE_URL, CHROMA_DB_PATH or QDRANT_URL",
                    }
    except Exception as e:
        checks["vector_db"] = {"status": "unhealthy", "error": str(e)}

    # Providers check (enhanced with detailed LLM testing)
    providers: list[dict[str, object]] = []
    try:
        providers_config = [
            {
                "name": "Aliyun",
                "env_keys": ["ALIYUN_MODEL_SERVER_KEY", "ALIYUN_API_KEY", "ALIYUN_KEY"],
                "base_url": os.getenv(
                    "ALIYUN_MODEL_SERVER_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                "is_primary": True,
            },
            {
                "name": "Azure OpenAI",
                "env_keys": ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY"],
                "base_url": os.getenv(
                    "AZURE_OPENAI_ENDPOINT", "https://{resource}.openai.azure.com"
                ),
                "is_primary": True,
            },
            {
                "name": "Ollama (GCP)",
                "env_keys": ["LOCAL_LLM_API_KEY", "GCP_LLM_API_KEY"],
                "base_url": os.getenv("OLLAMA_GCP_URL", os.getenv("OLLAMA_GCP_BASE_URL", "")),
                "is_primary": True,
            },
            {
                "name": "LlamaCpp (GCP)",
                "env_keys": ["LOCAL_LLM_API_KEY", "GCP_LLM_API_KEY"],
                "base_url": os.getenv(
                    "LLAMACPP_GCP_URL", os.getenv("LLAMACPP_GCP_BASE_URL", "")
                ),
                "is_primary": True,
            },
            {
                "name": "Ollama (Kamatera)",
                "env_key": "KAMATERA_LLM_API_KEY",
                "base_url": os.getenv("KAMATERA_LLM_URL", "http://66.55.77.147:8000"),
                "is_primary": False,
            },
            {
                "name": "Ollama (Local)",
                "env_key": "OLLAMA_API_KEY",
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "is_primary": False,
            },
            {
                "name": "Anthropic",
                "env_key": "ANTHROPIC_API_KEY",
                "base_url": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                "is_primary": False,
            },
            {
                "name": "OpenAI",
                "env_key": "OPENAI_API_KEY",
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
                "is_primary": False,
            },
            {
                "name": "Groq",
                "env_key": "GROQ_API_KEY",
                "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com"),
                "is_primary": False,
            },
            {
                "name": "DeepSeek",
                "env_key": "DEEPSEEK_API_KEY",
                "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                "is_primary": False,
            },
            {
                "name": "Gemini",
                "env_key": "GEMINI_API_KEY",
                "base_url": os.getenv("GEMINI_BASE_URL", "https://generative.googleapis.com"),
                "is_primary": False,
            },
        ]

        for p in providers_config:
            # Allow explicit disabling per-provider via <PROVIDER>_ENABLED env var (false/0/no -> disabled)
            enabled_override = os.getenv(
                f"{p['name'].upper().replace(' ', '_').replace('(', '').replace(')', '')}_ENABLED"
            )
            if enabled_override is not None and enabled_override.lower() in (
                "0",
                "false",
                "no",
            ):
                result = {
                    "enabled": False,
                    "is_primary": p.get("is_primary", False),
                }
                providers.append({p["name"]: result})
                continue

            key = None
            env_keys = p.get("env_keys") or []
            if isinstance(env_keys, list) and env_keys:
                for env_name in env_keys:
                    value = os.getenv(str(env_name))
                    if value:
                        key = value
                        break
            if not key:
                key = os.getenv(p["env_key"]) if p.get("env_key") else None
            result = {
                "enabled": bool(key),
                "is_primary": p.get("is_primary", False),
            }

            if key:
                try:
                    # Enhanced LLM provider health check
                    provider_health = await check_llm_provider_health(
                        p["name"].split()[0],
                        key,
                        p["base_url"],
                    )
                    result.update(provider_health)

                    # Mark as healthy only if both connectivity and inference work
                    if provider_health.get("status") == "healthy":
                        result["status"] = "healthy"
                    elif provider_health.get("status") == "degraded":
                        result["status"] = "degraded"
                    else:
                        result["status"] = "unhealthy"

                except Exception as e:
                    result["status"] = "unhealthy"
                    result["error"] = str(e)
            else:
                result["status"] = "disabled"

            providers.append({p["name"]: result})

        checks["providers"] = providers
    except Exception as e:
        checks["providers"] = {"status": "unhealthy", "error": str(e)}

    # App-level check
    overall = "healthy"
    for _, v in checks.items():
        if isinstance(v, dict) and v.get("status") == "unhealthy":
            overall = "degraded"
        if isinstance(v, list):
            for item in v:
                # item is like {"Anthropic": {...}}
                for _, s in item.items():
                    if s.get("status") == "unreachable":
                        overall = "degraded"

    return HealthCheckResponse(status=overall, checks=checks)


__all__ = ["router", "HealthCheckResponse", "check_database_health"]
