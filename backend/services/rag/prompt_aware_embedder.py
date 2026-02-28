from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence, overload


def _with_prefix(text: str, prefix: str) -> str:
    if not prefix:
        return text
    return f"{prefix}{text}"


def _with_instruction(text: str, instruction_prefix: str) -> str:
    if not instruction_prefix:
        return text
    inst = instruction_prefix
    if inst and not inst.endswith(" "):
        inst += " "
    return f"{inst}{text}"


def _to_list(vec: Any) -> list[float]:
    """Best-effort conversion of embedding vectors to a Python list[float]."""
    if vec is None:
        return []
    if hasattr(vec, "tolist"):
        return vec.tolist()
    # vec may already be a list/tuple/iterable of numbers.
    return list(vec)


@dataclass(frozen=True)
class PromptAwareConfig:
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    instruction_prefix: str = ""
    normalize_embeddings: bool = True


class PromptAwareEmbedder:
    """Wrap a sentence-transformers style embedder with query/passage prompting.

    The wrapped object must expose a `encode(texts, **kwargs)` method compatible
    with sentence-transformers.
    """

    def __init__(self, base_embedder: Any, config: PromptAwareConfig | None = None):
        self.base_embedder = base_embedder
        self.config = config or PromptAwareConfig()

    def _format(self, text: str, *, kind: str) -> str:
        prefix = self.config.query_prefix if kind == "query" else self.config.passage_prefix
        return _with_instruction(_with_prefix(text, prefix), self.config.instruction_prefix)

    def encode_query(self, texts: str | Sequence[str], **kwargs: Any):
        return self._encode(texts, kind="query", **kwargs)

    def encode_passage(self, texts: str | Sequence[str], **kwargs: Any):
        return self._encode(texts, kind="passage", **kwargs)

    def encode(self, texts: str | Sequence[str], **kwargs: Any):
        # Backward-compatible default: treat unknown calls as passage encoding.
        return self.encode_passage(texts, **kwargs)

    def _encode(self, texts: str | Sequence[str], *, kind: str, **kwargs: Any):
        if isinstance(texts, str):
            payload: str | list[str] = self._format(texts, kind=kind)
        else:
            payload = [self._format(t, kind=kind) for t in texts]

        # Ensure we default to normalized embeddings when supported.
        if "normalize_embeddings" not in kwargs:
            kwargs["normalize_embeddings"] = self.config.normalize_embeddings

        return self.base_embedder.encode(payload, **kwargs)


__all__ = [
    "PromptAwareEmbedder",
    "PromptAwareConfig",
    "_to_list",
]
