from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GatewayCheckResult:
    allowed: bool
    intent: Any
    estimated_tokens: Optional[int]
    risk_score: Optional[float]
    fallback_level: Optional[str] = None
    retry_after: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class ProviderConfig:
    name: str
    api_key: str = ""
    base_url: str = ""
    timeout: int = 30
    retries: int = 2
    cost_per_token_input: float = 0.0
    cost_per_token_output: float = 0.0
    latency_threshold_ms: int = 5000

    @staticmethod
    def from_dict(name: str, cfg: Dict[str, Any]) -> "ProviderConfig":
        return ProviderConfig(
            name=name,
            api_key=cfg.get("api_key", ""),
            base_url=cfg.get("base_url", ""),
            timeout=cfg.get("timeout", 30),
            retries=cfg.get("retries", cfg.get("max_retries", 2)),
            cost_per_token_input=cfg.get("cost_per_token_input", 0.0),
            cost_per_token_output=cfg.get("cost_per_token_output", 0.0),
            latency_threshold_ms=cfg.get("latency_threshold_ms", 5000),
        )


@dataclass
class ProviderResult:
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    latency_ms: Optional[float] = None


__all__ = ["GatewayCheckResult", "ProviderConfig", "ProviderResult"]
