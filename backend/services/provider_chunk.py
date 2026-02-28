from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field


class ChunkUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ProviderChunk(BaseModel):
    """Internal streaming chunk contract across providers."""

    content: str = ""
    role: str = "assistant"
    finish_reason: str | None = None
    usage: ChunkUsage | None = None
    cost: float | None = Field(default=None, ge=0)

    @classmethod
    def from_mapping(cls, value: Any) -> "ProviderChunk":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return cls(content=str(value))

        usage_raw = value.get("usage")
        usage: ChunkUsage | None = None
        if isinstance(usage_raw, Mapping):
            prompt_tokens = int(usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)) or 0)
            completion_tokens = int(
                usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)) or 0
            )
            total_tokens = int(
                usage_raw.get("total_tokens", prompt_tokens + completion_tokens) or 0
            )
            usage = ChunkUsage(
                prompt_tokens=max(0, prompt_tokens),
                completion_tokens=max(0, completion_tokens),
                total_tokens=max(0, total_tokens),
            )

        return cls(
            content=str(value.get("content", "") or ""),
            role=str(value.get("role", "assistant") or "assistant"),
            finish_reason=(
                None
                if value.get("finish_reason") is None
                else str(value.get("finish_reason"))
            ),
            usage=usage,
            cost=(None if value.get("cost") is None else float(value.get("cost"))),
        )


__all__ = ["ChunkUsage", "ProviderChunk"]
