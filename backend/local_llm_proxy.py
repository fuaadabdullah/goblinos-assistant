#!/usr/bin/env python3
"""
Local LLM Proxy Service
FastAPI proxy that enforces x-api-key authentication and routes requests
to local Ollama and llama.cpp servers (self-hosted/GCP).

This runs as a separate service from the main Goblin Assistant backend,
providing secure access to local LLMs with API key validation.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add RAG service import
try:
    from .services.rag_service import RAGService
    from .config import settings

    RAG_AVAILABLE = True
except ImportError:
    try:
        from services.rag_service import RAGService
        from config import settings

        RAG_AVAILABLE = True
    except ImportError:
        RAG_AVAILABLE = False
        logging.warning("RAG service not available - install chromadb and sentence-transformers")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
# Read API key (empty string disables auth for local dev/testing)
def _current_api_key() -> str:
    return os.getenv("LOCAL_LLM_API_KEY", "")


# Allow overriding base URLs
OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL", os.getenv("LOCAL_LLM_PROXY_OLLAMA_URL", "http://localhost:11434")
)
LLAMACPP_URL = os.getenv("LLAMACPP_URL", "http://localhost:8080")

# Routing keywords
CHEAP_FALLBACK_MODEL = "goblin-simple-llama-1b"
OLLAMA_KEYWORDS = ("phi3", "gemma", "qwen", "deepseek", CHEAP_FALLBACK_MODEL)
LLAMACPP_KEYWORDS = ("llama", "gguf", "active")


def _parse_float_env(name: str, default: float, *, minimum: Optional[float] = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default", name, raw)
        return default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _parse_int_env(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default", name, raw)
        return default
    if minimum is not None:
        value = max(minimum, value)
    return value


# HTTP client with timeout
_connect_timeout = _parse_float_env("LOCAL_LLM_PROXY_CONNECT_TIMEOUT_S", 10.0, minimum=0.5)
_read_timeout = _parse_float_env("LOCAL_LLM_PROXY_TIMEOUT_S", 300.0, minimum=5.0)
_timeout = httpx.Timeout(_read_timeout, connect=_connect_timeout)
_limits = httpx.Limits(max_keepalive_connections=10, max_connections=50)
client = httpx.AsyncClient(timeout=_timeout, limits=_limits)  # 5 minute timeout for LLM requests


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await client.aclose()


app = FastAPI(title="Local LLM Proxy", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    model: str
    messages: list
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


def _is_authorized(request: Request) -> bool:
    """Return True if request contains the valid API key header or bearer token.

    Behavior:
    - If API_KEY is empty, treat service as dev mode and accept all requests.
    - Otherwise, accept either 'x-api-key' header or 'Authorization: Bearer <token>'.
    """
    api_key = _current_api_key()
    if not api_key:
        return True
    # check x-api-key
    x_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if x_key and x_key == api_key:
        return True
    # check bearer token
    auth = request.headers.get("authorization", "")
    if auth and auth.startswith("Bearer "):
        token = auth.replace("Bearer ", "", 1).strip()
        if token == api_key:
            return True
    return False


def require_auth(request: Request) -> None:
    if not _is_authorized(request):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _auth_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = _current_api_key()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


async def _get_ollama_models() -> list[str]:
    try:
        response = await client.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except Exception as e:
        logger.error(f"Ollama models error: {e}")
    return []


def _get_llamacpp_models() -> list[str]:
    # llama.cpp doesn't have a standard models endpoint, so return configured models
    return ["active-model"]


def _apply_fallback_mode(body: Dict[str, Any], model: str) -> tuple[Dict[str, Any], str]:
    is_fallback = body.get("fallback_mode", False)
    if is_fallback or model == CHEAP_FALLBACK_MODEL:
        logger.info(f"Using cheap fallback model for request (fallback_mode: {is_fallback})")
        body["model"] = CHEAP_FALLBACK_MODEL
        model = CHEAP_FALLBACK_MODEL
        body["max_tokens"] = min(body.get("max_tokens", 512), 256)
        body["temperature"] = max(body.get("temperature", 0.7), 0.8)

        messages = body.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (
                "You are a basic AI assistant in fallback mode. Keep responses brief. "
                + messages[0]["content"]
            )
        else:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a basic AI assistant in fallback mode due to high load. Keep responses brief and to the point.",
                },
            )
        body["messages"] = messages

    return body, model


def _resolve_chat_target(model: str, body: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")
    if any(keyword in model for keyword in OLLAMA_KEYWORDS):
        return f"{OLLAMA_URL}/v1/chat/completions", body
    if any(keyword in model for keyword in LLAMACPP_KEYWORDS):
        return f"{LLAMACPP_URL}/completion", convert_openai_to_llamacpp(body)
    raise HTTPException(status_code=400, detail=f"Unknown model: {model}")


async def _request_with_retries(method: str, url: str, *, json: Dict[str, Any]) -> httpx.Response:
    max_retries = _parse_int_env("LOCAL_LLM_PROXY_MAX_RETRIES", 2, minimum=0)
    backoff = _parse_float_env("LOCAL_LLM_PROXY_BACKOFF_S", 0.5, minimum=0.0)
    for attempt in range(max_retries + 1):
        try:
            response = await client.request(method, url, json=json, headers=_auth_headers())
        except httpx.TimeoutException:
            if attempt < max_retries:
                await asyncio.sleep(backoff * (2**attempt))
                continue
            raise
        if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            await response.aclose()
            await asyncio.sleep(backoff * (2**attempt))
            continue
        return response
    return response


async def _forward_json_request(url: str, body: Dict[str, Any]) -> Response:
    response = await _request_with_retries("POST", url, json=body)

    if response.status_code != 200:
        logger.error("LLM request failed: %s %s", response.status_code, response.text)
        raise HTTPException(status_code=response.status_code, detail="LLM request failed")

    return Response(content=response.content, media_type=response.headers.get("content-type"))


def _build_openai_model_list(models: Dict[str, list[str]]) -> dict:
    data = []
    for m in models.get("ollama", []):
        data.append({"id": m, "object": "model", "created": None, "owned_by": "ollama"})
    for m in models.get("llamacpp", []):
        data.append({"id": m, "object": "model", "created": None, "owned_by": "llamacpp"})
    return {"object": "list", "data": data}


def _get_rag_service():
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG service not available")
    return RAGService(
        enable_enhanced=settings.enable_enhanced_rag,
        chroma_path=settings.rag_chroma_path,
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(f"{request.method} {request.url}")
    response = await call_next(request)
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "local-llm-proxy"}


@app.get("/models")
async def list_models(request: Request):
    """List available models from both Ollama and llama.cpp (legacy endpoint)."""
    require_auth(request)

    models = {
        "ollama": await _get_ollama_models(),
        "llamacpp": _get_llamacpp_models(),
    }

    return {"models": models}


@app.get("/api/status")
async def api_status(request: Request):
    """Native API status endpoint for Ollama compatibility"""
    # no auth required for basic status
    return {"status": "ok", "name": "local-llm-proxy", "uptime": 0}


@app.get("/api/tags")
async def api_tags(request: Request):
    """Ollama tags endpoint to list models in native format"""
    require_auth(request)
    try:
        response = await client.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            return {"models": response.json().get("models", [])}
    except Exception as e:
        logger.error(f"api_tags failed: {e}")
    return {"models": []}


@app.get("/v1/models")
async def openai_list_models(request: Request):
    """OpenAI-compatible list models endpoint.

    Produces a JSON response with `object: list` and `data: [ {id, object, ...} ]`.
    """
    require_auth(request)
    models = {
        "ollama": await _get_ollama_models(),
        "llamacpp": _get_llamacpp_models(),
    }
    return _build_openai_model_list(models)


@app.post("/api/chat")
async def ollama_chat(request: Request):
    """Direct proxy to Ollama /api/chat endpoint (native)."""
    require_auth(request)

    try:
        body = await request.json()
        return await _forward_json_request(f"{OLLAMA_URL}/api/chat", body)

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama request timeout")
    except Exception as e:
        logger.error(f"Ollama chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/chat/completions")
async def chat_completions(request: Request):
    """Route chat completions to appropriate local LLM with fallback support"""
    require_auth(request)

    try:
        body = await request.json()
        model = body.get("model", "").lower()
        body, model = _apply_fallback_mode(body, model)
        target_url, payload = _resolve_chat_target(model, body)
        return await _forward_json_request(target_url, payload)

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timeout")
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def convert_openai_to_llamacpp(openai_body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI chat format to llama.cpp completion format"""
    messages = openai_body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Extract the last user message as prompt
    prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            prompt = msg.get("content", "")
            break

    if not prompt:
        raise HTTPException(status_code=400, detail="No user message found")

    # Build llama.cpp format
    llamacpp_body = {
        "prompt": prompt,
        "n_predict": openai_body.get("max_tokens", 512),
        "temperature": openai_body.get("temperature", 0.7),
        "stop": ["\n\n", "###"],  # Common stop sequences
        "stream": openai_body.get("stream", False),
    }

    return llamacpp_body


@app.post("/api/generate")
async def api_generate(request: Request):
    """Native Ollama generate endpoint that may return streaming/cumulative output.

    We'll call the Ollama `/api/generate` endpoint directly and forward the result.
    """
    require_auth(request)
    body = await request.json()

    try:
        return await _forward_json_request(f"{OLLAMA_URL}/api/generate", body)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama generate timed out")
    except Exception as e:
        logger.error(f"api_generate error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/generate")
async def v1_generate(request: Request):
    """Raptor (Raptor Mini) compatible /v1/generate endpoint.
    Returns `{ok: True, result: {response: ...}}` format.
    """
    require_auth(request)
    body = await request.json()
    # Try to call raptor endpoint directly (assumed to expose /v1/generate)
    try:
        return await _forward_json_request(f"{LLAMACPP_URL}/v1/generate", body)
    except Exception:
        # Fallback: return an emulated response if the raptor service is not available
        prompt = body.get("prompt", "")
        mock_text = f"(Local Raptor) Mock response for prompt: {prompt[:200]}"
        return {"ok": True, "result": {"response": mock_text}}


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint — forwards to Ollama or llama.cpp depending on model.
    This is a convenience wrapper for clients using OpenAI-compatible API shape.
    """
    require_auth(request)
    body = await request.json()
    model = body.get("model", "").lower()
    body, model = _apply_fallback_mode(body, model)
    target_url, payload = _resolve_chat_target(model, body)
    return await _forward_json_request(target_url, payload)


# RAG Endpoints
class DocumentRequest(BaseModel):
    content: str
    id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    collection: Optional[str] = "documents"


class RAGQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None
    collection: Optional[str] = "documents"


@app.post("/rag/documents")
async def add_documents(request: Request, doc_request: DocumentRequest):
    """Add documents to the RAG vector database."""
    require_auth(request)

    try:
        rag_service = _get_rag_service()
        documents = [
            {
                "content": doc_request.content,
                "id": doc_request.id,
                "metadata": doc_request.metadata or {},
            }
        ]

        success = await rag_service.add_documents(documents, doc_request.collection)

        if success:
            return {
                "status": "success",
                "message": f"Added document to {doc_request.collection}",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add document")

    except Exception as e:
        logger.error(f"RAG document addition failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/rag/query")
async def rag_query(request: Request, query_request: RAGQueryRequest):
    """Perform RAG query with dense retrieval and context generation."""
    require_auth(request)

    try:
        rag_service = _get_rag_service()

        # Run complete RAG pipeline (enhanced if enabled)
        if settings.enable_enhanced_rag:
            result = await rag_service.enhanced_rag_pipeline(
                query=query_request.query,
                session_id=query_request.session_id,
                filters=query_request.filters,
            )
        else:
            result = await rag_service.rag_pipeline(
                query=query_request.query,
                session_id=query_request.session_id,
                filters=query_request.filters,
            )

        return result

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/rag/health")
async def rag_health(request: Request):
    """Check RAG service health."""
    require_auth(request)

    if not RAG_AVAILABLE:
        return {"status": "unavailable", "message": "RAG dependencies not installed"}

    try:
        rag_service = _get_rag_service()
        # Simple health check - try to get collection count
        collections = rag_service.chroma_client.list_collections()
        return {
            "status": "healthy",
            "collections": len(collections),
            "service": "rag",
            "enhanced_rag_enabled": settings.enable_enhanced_rag,
        }
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
