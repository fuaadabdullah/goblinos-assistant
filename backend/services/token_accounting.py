"""
Token accounting utilities with pluggable tokenizers.

Uses provider tokenizers when available (tiktoken), and falls back to a
lightweight heuristic tokenizer when not.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import tiktoken  # type: ignore

    TIKTOKEN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    TIKTOKEN_AVAILABLE = False
    tiktoken = None
    logger.info("tiktoken not available: %s", exc)


class TokenAccountingService:
    """Centralized token counting and chunking."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        force_fallback: bool = False,
    ) -> None:
        self.model = model
        self.provider = provider
        self.force_fallback = force_fallback
        self._encoding = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenize(text))

    def tokenize(self, text: str) -> List[str] | List[int]:
        if not text:
            return []

        encoding = self._get_encoding()
        if encoding:
            return encoding.encode(text)

        # Fallback: treat whitespace-separated spans as tokens.
        return re.findall(r"\S+", text)

    def detokenize(self, tokens: Sequence[str] | Sequence[int]) -> str:
        if not tokens:
            return ""

        encoding = self._get_encoding()
        if encoding:
            return encoding.decode(list(tokens))

        return " ".join(tokens)  # type: ignore[arg-type]

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        if not text:
            return []
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        tokens = self.tokenize(text)
        if not tokens:
            return []

        step = max(1, chunk_size - max(0, overlap))
        chunks: List[str] = []
        i = 0

        while i < len(tokens):
            chunk_tokens = tokens[i : min(i + chunk_size, len(tokens))]
            chunks.append(self.detokenize(chunk_tokens))

            if i + chunk_size >= len(tokens):
                break
            i += step

        return chunks

    def _get_encoding(self):
        if self.force_fallback or not TIKTOKEN_AVAILABLE:
            return None

        if self._encoding is not None:
            return self._encoding

        try:
            if self.model:
                self._encoding = tiktoken.encoding_for_model(self.model)
            else:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.info("tiktoken encoding unavailable: %s", exc)
            self._encoding = None

        return self._encoding
