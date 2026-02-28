"""Cost estimation utilities for LLM token usage.

Provides helpers to normalise heterogeneous usage dicts from different
providers into a common ``{input_tokens, output_tokens, total_tokens}``
shape and to compute a USD cost estimate.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def normalize_usage(raw: Any) -> dict[str, int]:
    """Normalise a provider's raw usage dict into a standard shape."""
    if not isinstance(raw, dict):
        return {}

    def _to_int(value: Any) -> int:
        try:
            n = int(value)
            return max(0, n)
        except Exception:
            return 0

    input_tokens = _to_int(
        raw.get("input_tokens")
        or raw.get("prompt_tokens")
        or raw.get("prompt_eval_count")
        or raw.get("promptTokenCount")
    )
    output_tokens = _to_int(
        raw.get("output_tokens")
        or raw.get("completion_tokens")
        or raw.get("eval_count")
        or raw.get("candidatesTokenCount")
    )
    total_tokens = _to_int(raw.get("total_tokens") or raw.get("totalTokenCount"))
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def get_usd_per_1k(provider_id: str) -> tuple[float, float]:
    """Return ``(input_cost_per_1k, output_cost_per_1k)`` in USD."""
    # Self-hosted inference is treated as "free" in the UI.
    if provider_id in {"ollama_gcp", "llamacpp_gcp", "aliyun"}:
        return (0.0, 0.0)

    try:
        # Prefer the centralized ProviderConfig costs if available.
        from ..providers.provider_registry import (
            get_provider_registry as _get_cost_registry,
        )

        cfg = _get_cost_registry().get_provider(provider_id)
        if cfg is not None:
            return (
                float(cfg.cost_per_token_input or 0.0),
                float(cfg.cost_per_token_output or 0.0),
            )
    except Exception:
        pass

    # Fallback: conservative defaults (USD per 1k tokens).
    fallback: dict[str, tuple[float, float]] = {
        "openai": (0.002, 0.006),
        "azure_openai": (0.002, 0.006),
        "anthropic": (0.008, 0.024),
        "openrouter": (0.003, 0.009),
        "groq": (0.0002, 0.0002),
        "deepseek": (0.0002, 0.0004),
        "gemini": (0.0005, 0.001),
        "siliconeflow": (0.001, 0.002),
    }
    return fallback.get(provider_id, (0.02, 0.02))


def compute_cost_usd(usage: dict[str, int], provider_id: str) -> float:
    """Estimate the USD cost of a completion based on token usage."""
    try:
        input_rate, output_rate = get_usd_per_1k(provider_id)
        it = int(usage.get("input_tokens") or 0)
        ot = int(usage.get("output_tokens") or 0)
        if not it and not ot:
            total = int(usage.get("total_tokens") or 0)
            it = int(total * 0.4)
            ot = max(0, total - it)
        cost = (it / 1000.0) * float(input_rate) + (ot / 1000.0) * float(
            output_rate
        )
        return float(round(cost, 6))
    except Exception:
        return 0.0
