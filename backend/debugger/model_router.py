from typing import Dict, Any, Optional
import os
import httpx
from dataclasses import dataclass

_RAPTOR_URL = os.getenv("RAPTOR_URL")
_RAPTOR_KEY = os.getenv("RAPTOR_API_KEY")
_FALLBACK_URL = os.getenv("FALLBACK_MODEL_URL")
_FALLBACK_KEY = os.getenv("FALLBACK_MODEL_KEY")

RAPTOR_TASKS = {"summarize_trace", "quick_fix", "unit_test_hint", "infer_function_name"}


@dataclass
class ModelRoute:
    url: str
    api_key: Optional[str]
    model_name: str


class ModelRouter:
    def __init__(
        self,
        raptor_url: Optional[str] = None,
        raptor_key: Optional[str] = None,
        fallback_url: Optional[str] = None,
        fallback_key: Optional[str] = None,
    ):
        # Allow explicit injection for tests; fall back to environment values at runtime
        self.raptor_url = (
            raptor_url if raptor_url is not None else os.getenv("RAPTOR_URL")
        )
        self.raptor_key = (
            raptor_key if raptor_key is not None else os.getenv("RAPTOR_API_KEY")
        )
        self.fallback_url = (
            fallback_url
            if fallback_url is not None
            else os.getenv("FALLBACK_MODEL_URL")
        )
        self.fallback_key = (
            fallback_key
            if fallback_key is not None
            else os.getenv("FALLBACK_MODEL_KEY")
        )

    def choose_model(self, task: str, context: Dict[str, Any]) -> ModelRoute:
        if task in RAPTOR_TASKS and self.raptor_url:
            return ModelRoute(
                url=self.raptor_url, api_key=self.raptor_key, model_name="raptor"
            )
        if self.fallback_url:
            return ModelRoute(
                url=self.fallback_url, api_key=self.fallback_key, model_name="fallback"
            )
        raise RuntimeError("No model endpoints configured")

    async def call_model(
        self, route: ModelRoute, payload: Dict[str, Any], timeout: int = 30
    ) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if route.api_key:
            headers["Authorization"] = f"Bearer {route.api_key}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(route.url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
