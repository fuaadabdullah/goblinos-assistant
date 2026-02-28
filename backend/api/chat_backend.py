"""
Fly.io Chat Backend - Lightweight LLM Service

Provides always-on chat capabilities using small quantized models.
Designed to run on Fly.io's shared-cpu-2x (2GB RAM) instances.

Models:
- TinyLlama-1.1B (default) - fits in 2GB with int4 quantization
- Phi-2 (2.7B) - optional, needs 4GB
- SmolLM (135M-1.7B) - ultra-lightweight option

Fallback chain:
1. Local llama.cpp inference (TinyLlama int4)
2. OpenAI API (if configured)
3. Error response
"""

import os
import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, AsyncGenerator, Dict, Any, List
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def _env_bool(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# Configuration
MODEL_ID = os.getenv("MODEL_ID", "Llama-3.2-1B-Instruct")
QUANTIZATION = os.getenv("QUANTIZATION", "int4")
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", "2048"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOBLIN_API_KEY = os.getenv("GOBLIN_API_KEY", "")
DISABLE_LOCAL_ML = _env_bool("DISABLE_LOCAL_ML", "false")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "DEFAULT_SYSTEM_PROMPT",
    "You are Goblin Assistant. Reply naturally and directly to the user. "
    "Do not explain what the user said. Do not provide meta-instructions. "
    "Be concise unless the user asks for more detail.",
)
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))

# Optional: automatically download a small GGUF model into the mounted volume.
# Default points to a public Llama 3.2 1B instruct quantization that fits in 2GB RAM.
MODEL_DOWNLOAD_URL = os.getenv(
    "MODEL_DOWNLOAD_URL",
    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
)
MODEL_FILENAME = os.getenv("MODEL_FILENAME", Path(MODEL_DOWNLOAD_URL).name)

_download_task: Optional[asyncio.Task] = None
_download_status: str = "idle"  # idle|downloading|complete|failed
_download_error: str = ""
_download_bytes: int = 0
_download_total_bytes: int = 0


@dataclass
class InferenceStats:
    """Track inference statistics."""

    total_requests: int = 0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    errors: int = 0
    local_requests: int = 0
    fallback_requests: int = 0


stats = InferenceStats()


# Optional llama-cpp-python for local inference
if DISABLE_LOCAL_ML:
    LLAMA_CPP_AVAILABLE = False
    logger.info("DISABLE_LOCAL_ML=true. Local llama.cpp inference is disabled.")
else:
    try:
        from llama_cpp import Llama

        LLAMA_CPP_AVAILABLE = True
    except ImportError:
        LLAMA_CPP_AVAILABLE = False
        logger.warning("llama-cpp-python not installed. Local inference disabled.")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(256, ge=1, le=2048)
    # If omitted, we use DEFAULT_TEMPERATURE (env-configurable). Using Optional
    # avoids silently defaulting to 0.7 which is too random for small models.
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    stream: bool = Field(False)
    model: Optional[str] = Field(None, description="Model override")


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    model: str
    engine: str
    stats: Dict[str, Any]


# Global model instance
_llm: Optional["Llama"] = None
_llm_model_path: Optional[Path] = None
_llm_lock = asyncio.Lock()  # llama.cpp context is not thread-safe


def get_model_path() -> Optional[Path]:
    """Get path to the quantized model file."""
    if MODEL_FILENAME:
        preferred = MODEL_DIR / MODEL_FILENAME
        if preferred.exists():
            return preferred

    # Check for GGUF model
    gguf_patterns = [
        f"*{QUANTIZATION}*.gguf",
        "*.gguf",
        "*Q4_K_M.gguf",
        "model.gguf",
    ]

    for pattern in gguf_patterns:
        matches = sorted(MODEL_DIR.glob(pattern))
        if matches:
            return matches[0]

    return None


def load_model(model_path: Optional[Path] = None) -> Optional["Llama"]:
    """Load the local LLM model."""
    global _llm

    if DISABLE_LOCAL_ML or not LLAMA_CPP_AVAILABLE:
        return None

    model_path = model_path or get_model_path()
    if not model_path or not model_path.exists():
        logger.warning(f"No model found in {MODEL_DIR}")
        return None

    try:
        logger.info(f"Loading model from {model_path}")
        _llm = Llama(
            model_path=str(model_path),
            n_ctx=MAX_CONTEXT,
            n_threads=2,  # Match Fly.io CPU count
            n_gpu_layers=0,  # CPU only on Fly.io
            verbose=False,
        )
        global _llm_model_path
        _llm_model_path = model_path
        logger.info("Model loaded successfully")
        return _llm
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


async def _download_model(url: str, dest_path: Path) -> bool:
    """Download a GGUF model file to disk (streamed). Returns True on success."""
    global _download_status, _download_error, _download_bytes, _download_total_bytes

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        if dest_path.exists() and dest_path.stat().st_size > 0:
            _download_status = "complete"
            _download_error = ""
            return True

        tmp_path = dest_path.with_suffix(dest_path.suffix + ".partial")
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

        _download_status = "downloading"
        _download_error = ""
        _download_bytes = 0
        _download_total_bytes = 0

        logger.info(f"Downloading model from {url} -> {dest_path}")

        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                try:
                    _download_total_bytes = int(resp.headers.get("content-length") or 0)
                except Exception:
                    _download_total_bytes = 0

                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        _download_bytes += len(chunk)

        tmp_path.replace(dest_path)
        _download_status = "complete"
        _download_error = ""
        logger.info("Model download complete")
        return True
    except Exception as e:
        _download_status = "failed"
        _download_error = f"{type(e).__name__}: {e}"
        logger.warning(f"Model download failed: {_download_error}")
        return False


async def _ensure_model_loaded() -> None:
    """Background task: download (if needed) and load the model."""
    if DISABLE_LOCAL_ML or not LLAMA_CPP_AVAILABLE:
        return

    if not MODEL_DOWNLOAD_URL:
        logger.warning("MODEL_DOWNLOAD_URL not set; cannot auto-download model.")
        return

    dest_path = MODEL_DIR / (MODEL_FILENAME or "model.gguf")

    # If the preferred model is already loaded, nothing to do.
    if _llm is not None and _llm_model_path == dest_path:
        return

    # Retry a few times to survive transient network issues.
    delay_s = 2.0
    for attempt in range(1, 6):
        if await _download_model(MODEL_DOWNLOAD_URL, dest_path):
            if load_model(dest_path):
                return
        logger.warning(f"Retrying model download in {delay_s:.0f}s (attempt {attempt}/5)")
        await asyncio.sleep(delay_s)
        delay_s = min(delay_s * 2.0, 60.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Goblin Chat Backend...")
    if DISABLE_LOCAL_ML:
        logger.info("Skipping local model bootstrap because DISABLE_LOCAL_ML=true.")
    else:
        # Don't block startup waiting for a large download. Load if present and
        # otherwise download+load in the background.
        global _download_task
        preferred = MODEL_DIR / MODEL_FILENAME if MODEL_FILENAME else None
        if preferred and preferred.exists():
            load_model(preferred)
        else:
            # Load any available model for best-effort service, while we download the preferred one.
            load_model()
        if _download_task is None and _llm is None and LLAMA_CPP_AVAILABLE:
            _download_task = asyncio.create_task(_ensure_model_loaded())
        elif _download_task is None and LLAMA_CPP_AVAILABLE:
            # A model is loaded, but we may still want to swap to the preferred model.
            _download_task = asyncio.create_task(_ensure_model_loaded())
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Goblin Chat Backend",
    description="Lightweight LLM inference service for Fly.io",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """Verify API key if configured."""
    if not GOBLIN_API_KEY:
        return True  # No auth configured

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    token = authorization.replace("Bearer ", "")
    if token != GOBLIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    engine = (
        "llama-cpp"
        if _llm
        else "fallback-openai"
        if OPENAI_API_KEY
        else "disabled"
        if DISABLE_LOCAL_ML
        else "none"
    )
    return HealthResponse(
        status="healthy" if (_llm or OPENAI_API_KEY) else "degraded",
        model=MODEL_ID,
        engine=engine,
        stats={
            "total_requests": stats.total_requests,
            "total_tokens": stats.total_tokens,
            "avg_latency_ms": round(stats.avg_latency_ms, 2),
            "errors": stats.errors,
            "local_requests": stats.local_requests,
            "fallback_requests": stats.fallback_requests,
            "model_path": str(get_model_path() or ""),
            "download_status": _download_status,
            "downloaded_bytes": _download_bytes,
            "total_bytes": _download_total_bytes,
            "download_error": _download_error,
        },
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Goblin Chat Backend",
        "version": "1.0.0",
        "model": MODEL_ID,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "health": "/health",
        },
    }


async def generate_local(
    messages: List[ChatMessage],
    max_tokens: int,
    temperature: Optional[float],
) -> Optional[str]:
    """Generate response using local llama.cpp model."""
    if not _llm:
        return None

    try:
        # Ensure deterministic, thread-safe access to the llama.cpp context.
        async with _llm_lock:
            _llm.reset()

            req_temp = float(temperature) if temperature is not None else DEFAULT_TEMPERATURE
            req_temp = max(0.0, min(req_temp, 1.5))
            req_max_tokens = int(max_tokens) if max_tokens is not None else 256
            req_max_tokens = max(1, min(req_max_tokens, 1024))

            chat_messages = [{"role": m.role, "content": m.content} for m in messages]

            response = await asyncio.to_thread(
                _llm.create_chat_completion,
                messages=chat_messages,
                max_tokens=req_max_tokens,
                temperature=req_temp,
                # Let the model decide where to stop; most GGUFs have an EOS token.
                stop=["</s>"],
            )

        choice = (response.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        return content.strip()
    except Exception as e:
        logger.error(f"Local inference error: {e}")
        return None


def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Format messages for TinyLlama chat template."""
    prompt_parts = []

    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"<|system|>\n{msg.content}</s>")
        elif msg.role == "user":
            prompt_parts.append(f"<|user|>\n{msg.content}</s>")
        elif msg.role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{msg.content}</s>")

    # Add assistant prompt for generation
    prompt_parts.append("<|assistant|>\n")

    return "\n".join(prompt_parts)


async def generate_openai_fallback(
    messages: List[ChatMessage],
    max_tokens: int,
    temperature: Optional[float],
) -> Optional[str]:
    """Fallback to OpenAI API."""
    if not OPENAI_API_KEY:
        return None

    try:
        req_temp = float(temperature) if temperature is not None else DEFAULT_TEMPERATURE
        req_temp = max(0.0, min(req_temp, 1.5))
        req_max_tokens = int(max_tokens) if max_tokens is not None else 256
        req_max_tokens = max(1, min(req_max_tokens, 1024))

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": m.role, "content": m.content} for m in messages
                    ],
                    "max_tokens": req_max_tokens,
                    "temperature": req_temp,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"OpenAI fallback error: {e}")
        return None


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    _: bool = Depends(verify_api_key),
):
    """OpenAI-compatible chat completion endpoint."""
    start_time = time.time()
    stats.total_requests += 1

    # Ensure we always have a system prompt to anchor the model.
    messages = list(request.messages or [])
    if not any(m.role == "system" for m in messages):
        messages.insert(0, ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))

    # Try local inference first
    content = await generate_local(
        messages,
        request.max_tokens,
        request.temperature,
    )

    if content:
        stats.local_requests += 1
    else:
        # Fallback to OpenAI
        content = await generate_openai_fallback(
            messages,
            request.max_tokens,
            request.temperature,
        )
        if content:
            stats.fallback_requests += 1

    if not content:
        stats.errors += 1
        raise HTTPException(status_code=503, detail="No inference backend available")

    # Update stats
    latency_ms = (time.time() - start_time) * 1000
    stats.avg_latency_ms = (
        stats.avg_latency_ms * (stats.total_requests - 1) + latency_ms
    ) / stats.total_requests

    # Estimate token count
    tokens = len(content.split()) + sum(
        len(m.content.split()) for m in messages
    )
    stats.total_tokens += tokens

    return ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model or MODEL_ID,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": sum(len(m.content.split()) for m in messages),
            "completion_tokens": len(content.split()),
            "total_tokens": tokens,
        },
    )


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "goblin",
                "permission": [],
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
