"""
Utility types for streaming responses.

Lightweight dataclasses/enums used by stream_processor and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class StreamEventType(str, Enum):
    CHUNK = "chunk"
    COMPLETION = "completion"
    ERROR = "error"


@dataclass
class StreamEvent:
    type: StreamEventType
    data: dict[str, Any]


@dataclass
class StreamMessage:
    role: str
    content: str


@dataclass
class StreamMetadata:
    provider: str | None = None
    model: str | None = None
    request_id: str | None = None


@dataclass
class StreamTokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class StreamChoice:
    index: int
    delta: StreamDelta | None = None
    message: StreamMessage | None = None
    finish_reason: StreamFinishReason | None = None


@dataclass
class StreamDelta:
    content: str
    role: str = "assistant"


class StreamFinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"


@dataclass
class StreamUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class StreamChunk:
    id: str
    object: str
    created: int
    model: str
    choices: list[StreamChoice]
    usage: StreamUsage | None = None


@dataclass
class StreamCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: list[StreamChoice]
    usage: StreamUsage


@dataclass
class StreamError:
    id: str
    object: str
    error: dict[str, Any]


@dataclass
class StreamResponse:
    data: dict[str, Any]
    metadata: StreamMetadata | None = None


@dataclass
class StreamState:
    session_id: str
    request_id: str
    user_id: str | None
    client_ip: str | None
    model: str
    temperature: float
    max_tokens: int | None
    start_time: float


@dataclass
class StreamStats:
    request_id: str
    start_time: float
    chunks_processed: int
    total_tokens: int
    response_time: float


__all__ = [
    "StreamEventType",
    "StreamEvent",
    "StreamMessage",
    "StreamMetadata",
    "StreamResponse",
    "StreamState",
    "StreamStats",
    "StreamTokenUsage",
    "StreamError",
    "StreamCompletion",
    "StreamChunk",
    "StreamChoice",
    "StreamUsage",
    "StreamDelta",
    "StreamFinishReason",
]
