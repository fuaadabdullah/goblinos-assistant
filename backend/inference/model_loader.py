"""Model Loader Module

Handles loading, caching, and managing ML models for inference.
Supports Ollama, vLLM, and direct HuggingFace model loading.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class ModelBackend(str, Enum):
    """Supported model backends"""
    OLLAMA = "ollama"
    VLLM = "vllm"
    HUGGINGFACE = "huggingface"
    LLAMA_CPP = "llama_cpp"


class ModelStatus(str, Enum):
    """Model loading status"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    backend: ModelBackend
    status: ModelStatus
    size_bytes: Optional[int] = None
    quantization: Optional[str] = None
    context_length: int = 4096
    loaded_at: Optional[float] = None
    last_used: Optional[float] = None
    gpu_memory_mb: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for model loading"""
    ollama_url: str = "http://localhost:11434"
    vllm_url: str = "http://localhost:8000"
    models_cache_dir: str = "/app/models"
    max_loaded_models: int = 3
    auto_unload_after_minutes: int = 30
    default_backend: ModelBackend = ModelBackend.OLLAMA
    
    # GPU settings
    gpu_memory_fraction: float = 0.9
    tensor_parallel_size: int = 1
    
    # Model-specific configs
    model_configs: dict[str, dict] = field(default_factory=dict)


class ModelLoader:
    """
    Unified model loader supporting multiple backends.
    
    Manages model lifecycle:
    - Loading models on demand
    - Caching loaded models
    - Auto-unloading unused models
    - Health checking loaded models
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self._loaded_models: dict[str, ModelInfo] = {}
        self._loading_locks: dict[str, asyncio.Lock] = {}
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the model loader and background tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("ModelLoader started")
    
    async def stop(self) -> None:
        """Stop the model loader and cleanup"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all clients
        for client in self._clients.values():
            if not client.is_closed:
                await client.aclose()
        
        logger.info("ModelLoader stopped")
    
    async def _get_client(self, backend: ModelBackend) -> httpx.AsyncClient:
        """Get HTTP client for backend"""
        if backend not in self._clients or self._clients[backend].is_closed:
            if backend == ModelBackend.OLLAMA:
                base_url = self.config.ollama_url
            elif backend == ModelBackend.VLLM:
                base_url = self.config.vllm_url
            else:
                base_url = ""
            
            self._clients[backend] = httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(300.0)  # Long timeout for model operations
            )
        
        return self._clients[backend]
    
    async def _cleanup_loop(self) -> None:
        """Background task to unload unused models"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_unused_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_unused_models(self) -> None:
        """Unload models that haven't been used recently"""
        current_time = time.time()
        threshold = self.config.auto_unload_after_minutes * 60
        
        models_to_unload = []
        for name, info in self._loaded_models.items():
            if info.status == ModelStatus.READY and info.last_used:
                if current_time - info.last_used > threshold:
                    models_to_unload.append(name)
        
        for name in models_to_unload:
            logger.info(f"Auto-unloading unused model: {name}")
            await self.unload_model(name)
    
    def get_loading_lock(self, model_name: str) -> asyncio.Lock:
        """Get or create a loading lock for a model"""
        if model_name not in self._loading_locks:
            self._loading_locks[model_name] = asyncio.Lock()
        return self._loading_locks[model_name]
    
    async def load_model(
        self,
        model_name: str,
        backend: Optional[ModelBackend] = None,
        force_reload: bool = False
    ) -> ModelInfo:
        """
        Load a model into memory.
        
        Args:
            model_name: Name of the model (e.g., "llama3.1:8b", "phi3:mini")
            backend: Backend to use (defaults to config default)
            force_reload: Force reload even if already loaded
            
        Returns:
            ModelInfo with status and metadata
        """
        backend = backend or self.config.default_backend
        
        # Check if already loaded
        if model_name in self._loaded_models and not force_reload:
            info = self._loaded_models[model_name]
            if info.status == ModelStatus.READY:
                info.last_used = time.time()
                return info
        
        # Use lock to prevent concurrent loading of same model
        async with self.get_loading_lock(model_name):
            # Double-check after acquiring lock
            if model_name in self._loaded_models and not force_reload:
                info = self._loaded_models[model_name]
                if info.status == ModelStatus.READY:
                    info.last_used = time.time()
                    return info
            
            # Check if we need to unload something first
            if len(self._loaded_models) >= self.config.max_loaded_models:
                await self._evict_lru_model()
            
            # Create info entry
            info = ModelInfo(
                name=model_name,
                backend=backend,
                status=ModelStatus.LOADING
            )
            self._loaded_models[model_name] = info
            
            try:
                if backend == ModelBackend.OLLAMA:
                    await self._load_ollama_model(model_name, info)
                elif backend == ModelBackend.VLLM:
                    await self._load_vllm_model(model_name, info)
                elif backend == ModelBackend.LLAMA_CPP:
                    await self._load_llama_cpp_model(model_name, info)
                else:
                    raise ValueError(f"Unsupported backend: {backend}")
                
                info.status = ModelStatus.READY
                info.loaded_at = time.time()
                info.last_used = time.time()
                
                logger.info(f"Model {model_name} loaded successfully on {backend}")
                return info
                
            except Exception as e:
                info.status = ModelStatus.ERROR
                info.metadata["error"] = str(e)
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
    
    async def _evict_lru_model(self) -> None:
        """Evict least recently used model"""
        if not self._loaded_models:
            return
        
        # Find LRU model
        lru_model = None
        lru_time = float("inf")
        
        for name, info in self._loaded_models.items():
            if info.status == ModelStatus.READY and info.last_used:
                if info.last_used < lru_time:
                    lru_time = info.last_used
                    lru_model = name
        
        if lru_model:
            logger.info(f"Evicting LRU model: {lru_model}")
            await self.unload_model(lru_model)
    
    async def _load_ollama_model(self, model_name: str, info: ModelInfo) -> None:
        """Load model using Ollama"""
        client = await self._get_client(ModelBackend.OLLAMA)
        
        # First check if model exists locally
        response = await client.get("/api/tags")
        if response.status_code == 200:
            data = response.json()
            local_models = {m.get("name"): m for m in data.get("models", [])}
            
            if model_name in local_models:
                model_data = local_models[model_name]
                info.size_bytes = model_data.get("size")
                info.metadata["digest"] = model_data.get("digest")
            else:
                # Need to pull the model
                logger.info(f"Pulling model {model_name} from Ollama registry...")
                await self._pull_ollama_model(model_name)
        
        # Warm up the model with a simple prompt
        response = await client.post(
            "/api/generate",
            json={
                "model": model_name,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1}
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to warm up model: {response.text}")
        
        # Get model info
        response = await client.post(
            "/api/show",
            json={"name": model_name}
        )
        
        if response.status_code == 200:
            data = response.json()
            info.context_length = data.get("parameters", {}).get("num_ctx", 4096)
            info.metadata["modelfile"] = data.get("modelfile")
            info.metadata["template"] = data.get("template")
    
    async def _pull_ollama_model(self, model_name: str) -> None:
        """Pull model from Ollama registry"""
        client = await self._get_client(ModelBackend.OLLAMA)
        
        response = await client.post(
            "/api/pull",
            json={"name": model_name, "stream": False},
            timeout=httpx.Timeout(1800.0)  # 30 minute timeout for large models
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to pull model: {response.text}")
    
    async def _load_vllm_model(self, model_name: str, info: ModelInfo) -> None:
        """Load model using vLLM (typically already loaded as server)"""
        client = await self._get_client(ModelBackend.VLLM)
        
        # vLLM models are typically loaded at server startup
        # We just verify the server is running and model is available
        response = await client.get("/v1/models")
        
        if response.status_code != 200:
            raise RuntimeError("vLLM server not available")
        
        data = response.json()
        available_models = [m.get("id") for m in data.get("data", [])]
        
        if model_name not in available_models:
            raise RuntimeError(f"Model {model_name} not loaded in vLLM server")
        
        info.metadata["vllm_available"] = True
    
    async def _load_llama_cpp_model(self, model_name: str, info: ModelInfo) -> None:
        """Load model using llama.cpp"""
        # For llama.cpp, we need the GGUF file path
        model_path = self._resolve_model_path(model_name)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        info.size_bytes = model_path.stat().st_size
        info.metadata["path"] = str(model_path)
        
        # Detect quantization from filename
        filename = model_path.name.lower()
        for quant in ["q4_k_m", "q5_k_m", "q8_0", "q4_0", "q4_1", "f16"]:
            if quant in filename:
                info.quantization = quant.upper()
                break
    
    def _resolve_model_path(self, model_name: str) -> Path:
        """Resolve model name to file path"""
        cache_dir = Path(self.config.models_cache_dir)
        
        # Check for model-specific config
        if model_name in self.config.model_configs:
            if "path" in self.config.model_configs[model_name]:
                return Path(self.config.model_configs[model_name]["path"])
        
        # Check common patterns
        patterns = [
            f"{model_name}.gguf",
            f"{model_name.replace(':', '-')}.gguf",
            f"{model_name.replace(':', '/')}.gguf",
        ]
        
        for pattern in patterns:
            path = cache_dir / pattern
            if path.exists():
                return path
        
        # Return default path (may not exist)
        return cache_dir / f"{model_name}.gguf"
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.
        
        Returns True if model was unloaded, False if not found.
        """
        if model_name not in self._loaded_models:
            return False
        
        info = self._loaded_models[model_name]
        info.status = ModelStatus.UNLOADING
        
        try:
            if info.backend == ModelBackend.OLLAMA:
                # Ollama doesn't have an explicit unload, but we can
                # load a tiny model to free memory
                pass
            
            del self._loaded_models[model_name]
            logger.info(f"Model {model_name} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            info.status = ModelStatus.ERROR
            return False
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a loaded model"""
        return self._loaded_models.get(model_name)
    
    async def list_loaded_models(self) -> list[ModelInfo]:
        """List all loaded models"""
        return list(self._loaded_models.values())
    
    async def is_model_ready(self, model_name: str) -> bool:
        """Check if a model is loaded and ready"""
        info = self._loaded_models.get(model_name)
        return info is not None and info.status == ModelStatus.READY
    
    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all backends"""
        results = {}
        
        # Check Ollama
        try:
            client = await self._get_client(ModelBackend.OLLAMA)
            response = await client.get("/api/tags", timeout=5.0)
            results["ollama"] = {
                "healthy": response.status_code == 200,
                "url": self.config.ollama_url
            }
        except Exception as e:
            results["ollama"] = {"healthy": False, "error": str(e)}
        
        # Check vLLM if configured
        if self.config.vllm_url:
            try:
                client = await self._get_client(ModelBackend.VLLM)
                response = await client.get("/v1/models", timeout=5.0)
                results["vllm"] = {
                    "healthy": response.status_code == 200,
                    "url": self.config.vllm_url
                }
            except Exception as e:
                results["vllm"] = {"healthy": False, "error": str(e)}
        
        # Add loaded models info
        results["loaded_models"] = {
            name: {
                "status": info.status.value,
                "backend": info.backend.value,
                "last_used": info.last_used
            }
            for name, info in self._loaded_models.items()
        }
        
        return results
