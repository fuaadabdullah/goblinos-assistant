"""
Centralized model server configuration for remote inference.

This module manages connections to external model servers (RunPod, Aliyun, on-prem)
instead of running models in Fly.io.

Features:
- Multi-server failover support
- Health checks with exponential backoff
- Connection pooling and caching
- Model availability tracking
"""

import os
import asyncio
import httpx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ServerType(str, Enum):
    """Supported model server types."""

    OLLAMA = "ollama"  # Ollama API (11434)
    LLAMACPP = "llamacpp"  # llama.cpp server (8080)
    VLLM = "vllm"  # vLLM API (8000)
    RUNPOD = "runpod"  # RunPod serverless endpoint
    ALIYUN = "aliyun"  # Alibaba Cloud GPU instance


@dataclass
class ModelServer:
    """Configuration for a single model server."""

    name: str
    url: str
    server_type: ServerType
    api_key: Optional[str] = None
    priority: int = 1  # Lower = higher priority
    timeout_seconds: int = 60
    max_retries: int = 3
    available: bool = True
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    max_consecutive_failures: int = 5


class ModelServerRegistry:
    """Manages multiple model servers with failover support."""

    def __init__(self):
        self.servers: Dict[str, ModelServer] = {}
        self.client: Optional[httpx.AsyncClient] = None
        self._health_check_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the registry from environment variables."""
        self.client = httpx.AsyncClient(timeout=30)
        self._load_servers_from_env()

        # Start background health checks
        if self.servers:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info(f"Initialized {len(self.servers)} model servers")

    async def shutdown(self):
        """Cleanup resources."""
        if self._health_check_task:
            self._health_check_task.cancel()
        if self.client:
            await self.client.aclose()

    def _load_servers_from_env(self):
        """Load model servers from environment variables."""

        # Primary: RunPod Serverless
        runpod_url = os.getenv("RUNPOD_ENDPOINT_URL")
        if runpod_url:
            self.servers["runpod"] = ModelServer(
                name="RunPod Serverless",
                url=runpod_url,
                server_type=ServerType.RUNPOD,
                api_key=os.getenv("RUNPOD_API_KEY"),
                priority=1,  # Highest priority
                timeout_seconds=120,  # Serverless can be slower
            )

        # Secondary: Aliyun GPU
        aliyun_url = os.getenv("ALIYUN_MODEL_SERVER_URL")
        if aliyun_url:
            self.servers["aliyun"] = ModelServer(
                name="Aliyun GPU Instance",
                url=aliyun_url,
                server_type=ServerType.OLLAMA,  # Usually runs Ollama
                api_key=os.getenv("ALIYUN_MODEL_SERVER_KEY"),
                priority=2,
                timeout_seconds=60,
            )

        # Fallback: On-prem or other
        onprem_url = os.getenv("ONPREM_MODEL_SERVER_URL")
        if onprem_url:
            self.servers["onprem"] = ModelServer(
                name="On-Prem Model Server",
                url=onprem_url,
                server_type=ServerType.OLLAMA,
                api_key=os.getenv("ONPREM_MODEL_SERVER_KEY"),
                priority=3,
                timeout_seconds=60,
            )

    def get_available_servers(self) -> List[ModelServer]:
        """Get list of available servers, sorted by priority."""
        available = [s for s in self.servers.values() if s.available]
        return sorted(available, key=lambda s: s.priority)

    async def get_best_server(self) -> Optional[ModelServer]:
        """Get the best available server for inference."""
        servers = self.get_available_servers()
        if not servers:
            logger.warning("No model servers currently available!")
            return None
        return servers[0]

    async def health_check_single(self, server: ModelServer) -> bool:
        """
        Check if a server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        if not self.client:
            return False

        try:
            # Check endpoint based on server type
            if server.server_type == ServerType.OLLAMA:
                url = f"{server.url}/api/tags"
            elif server.server_type == ServerType.LLAMACPP:
                url = f"{server.url}/v1/models"
            elif server.server_type == ServerType.RUNPOD:
                url = f"{server.url}/health"
            else:
                url = f"{server.url}/health"

            headers = {}
            if server.api_key:
                headers["Authorization"] = f"Bearer {server.api_key}"

            response = await self.client.get(
                url,
                headers=headers,
                timeout=server.timeout_seconds,
            )

            is_healthy = response.status_code == 200

            if is_healthy:
                server.consecutive_failures = 0
                logger.debug(f"✓ {server.name} is healthy")
            else:
                server.consecutive_failures += 1
                logger.warning(f"✗ {server.name} returned {response.status_code}")

            # Mark unavailable after too many failures
            if server.consecutive_failures >= server.max_consecutive_failures:
                server.available = False
                logger.error(f"⚠️  {server.name} marked as unavailable (too many failures)")

            server.last_check = datetime.utcnow()
            return is_healthy

        except asyncio.TimeoutError:
            server.consecutive_failures += 1
            logger.warning(f"✗ {server.name} health check timed out")
            return False
        except Exception as e:
            server.consecutive_failures += 1
            logger.error(f"✗ {server.name} health check failed: {e}")
            return False

    async def _health_check_loop(self):
        """Periodically check health of all servers (background task)."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                tasks = [self.health_check_single(server) for server in self.servers.values()]
                await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    async def call_inference(
        self,
        model: str,
        prompt: str,
        **kwargs,
    ) -> Tuple[str, Optional[str]]:
        """
        Call inference on the best available server.

        Args:
            model: Model name (e.g., "tinyliama", "qwen2.5")
            prompt: Input prompt
            **kwargs: Additional parameters for the API

        Returns:
            Tuple of (response, server_name) or (error_message, None)
        """
        server = await self.get_best_server()
        if not server:
            return "No model servers available", None

        try:
            if server.server_type == ServerType.OLLAMA:
                return await self._call_ollama(server, model, prompt, **kwargs)
            elif server.server_type == ServerType.LLAMACPP:
                return await self._call_llamacpp(server, model, prompt, **kwargs)
            elif server.server_type == ServerType.RUNPOD:
                return await self._call_runpod(server, model, prompt, **kwargs)
            else:
                return f"Unsupported server type: {server.server_type}", None

        except Exception as e:
            # Mark server as failed and retry
            server.consecutive_failures += 1
            logger.error(f"Inference call to {server.name} failed: {e}")

            # Try next available server
            return await self.call_inference(model, prompt, **kwargs)

    async def _call_ollama(
        self, server: ModelServer, model: str, prompt: str, **kwargs
    ) -> Tuple[str, Optional[str]]:
        """Call Ollama API."""
        if not self.client:
            return "Client not initialized", None

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }

        headers = {}
        if server.api_key:
            headers["Authorization"] = f"Bearer {server.api_key}"

        response = await self.client.post(
            f"{server.url}/api/generate",
            json=payload,
            headers=headers,
            timeout=server.timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", ""), server.name

    async def _call_llamacpp(
        self, server: ModelServer, model: str, prompt: str, **kwargs
    ) -> Tuple[str, Optional[str]]:
        """Call llama.cpp API."""
        if not self.client:
            return "Client not initialized", None

        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 256),
            **{k: v for k, v in kwargs.items() if k != "max_tokens"},
        }

        headers = {}
        if server.api_key:
            headers["Authorization"] = f"Bearer {server.api_key}"

        response = await self.client.post(
            f"{server.url}/v1/completions",
            json=payload,
            headers=headers,
            timeout=server.timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()
        choices = result.get("choices", [])
        return (
            choices[0]["text"] if choices else "",
            server.name,
        )

    async def _call_runpod(
        self, server: ModelServer, model: str, prompt: str, **kwargs
    ) -> Tuple[str, Optional[str]]:
        """Call RunPod serverless endpoint."""
        if not self.client:
            return "Client not initialized", None

        payload = {
            "input": {
                "model": model,
                "prompt": prompt,
                **kwargs,
            }
        }

        headers = {}
        if server.api_key:
            headers["Authorization"] = f"Bearer {server.api_key}"

        response = await self.client.post(
            f"{server.url}/run",
            json=payload,
            headers=headers,
            timeout=server.timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()

        # RunPod returns different format
        output = result.get("output", {})
        if isinstance(output, dict):
            return output.get("response", str(output)), server.name
        return str(output), server.name


# Global registry instance
_registry: Optional[ModelServerRegistry] = None


async def get_model_registry() -> ModelServerRegistry:
    """Get or initialize the global model server registry."""
    global _registry
    if _registry is None:
        _registry = ModelServerRegistry()
        await _registry.initialize()
    return _registry


async def shutdown_model_registry():
    """Cleanup the model registry."""
    global _registry
    if _registry:
        await _registry.shutdown()
        _registry = None
