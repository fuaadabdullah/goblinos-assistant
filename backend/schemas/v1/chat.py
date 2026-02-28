from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="Role of the message sender (user, assistant, system)"
    )
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    model: Optional[str] = Field(
        None,
        description="Specific model to use (optional, auto-routed if not provided)",
    )
    intent: Optional[str] = Field(
        None, description="Explicit intent (code-gen, creative, rag, chat, etc.)"
    )
    latency_target: Optional[str] = Field(
        "medium", description="Latency requirement (ultra_low, low, medium, high)"
    )
    context: Optional[str] = Field(
        None, description="Additional context for RAG/retrieval"
    )
    cost_priority: Optional[bool] = Field(
        False, description="Prioritize cost over quality"
    )
    stream: Optional[bool] = Field(False, description="Stream the response")
    temperature: Optional[float] = Field(None, description="Override temperature")
    max_tokens: Optional[int] = Field(None, description="Override max tokens")
    top_p: Optional[float] = Field(None, description="Override top_p")
    enable_verification: Optional[bool] = Field(
        True, description="Enable output safety verification (default: True)"
    )
    enable_confidence_scoring: Optional[bool] = Field(
        True, description="Enable confidence scoring and escalation (default: True)"
    )
    auto_escalate: Optional[bool] = Field(
        True,
        description="Automatically escalate to better model if confidence is low (default: True)",
    )
    sla_target_ms: Optional[float] = Field(
        None, description="SLA target response time in milliseconds"
    )
    cost_budget: Optional[float] = Field(
        None, description="Maximum cost per request in USD"
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature_field(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 2):
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens_field(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 4096):
            raise ValueError("Max tokens must be between 1 and 4096")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p_field(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Top_p must be between 0 and 1")
        return v

    @field_validator("latency_target")
    @classmethod
    def validate_latency_target_field(cls, v: Optional[str]) -> Optional[str]:
        valid_targets = ["ultra_low", "low", "medium", "high"]
        if v is not None and v not in valid_targets:
            raise ValueError(
                f"Latency target must be one of: {', '.join(valid_targets)}"
            )
        return v

    @field_validator("intent")
    @classmethod
    def validate_intent_field(cls, v: Optional[str]) -> Optional[str]:
        valid_intents = [
            "code-gen",
            "creative",
            "explain",
            "summarize",
            "rag",
            "retrieval",
            "chat",
            "classification",
            "status",
            "translation",
            "analyze",
            "solve",
            "reason",
        ]
        if v is not None and v not in valid_intents:
            raise ValueError(f"Intent must be one of: {', '.join(valid_intents)}")
        return v


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    metadata: Optional[Dict[str, Any]] = None

