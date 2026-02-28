from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GenerateMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class GenerateRequest(BaseModel):
    messages: List[GenerateMessage] = Field(..., description="Structured chat messages")
    model: str = Field("llama2", description="Model hint for upstream providers")
    provider: Optional[str] = Field(
        None, description="Optional provider hint (e.g. ollama_gcp, llamacpp_gcp)"
    )
    max_tokens: Optional[int] = Field(None, ge=1, le=2048)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)

