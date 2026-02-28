"""
Ranking utilities for RAG retrieval results.
Handles reranking, filtering, and token limit trimming.
"""

import logging
from typing import List, Dict, Any, Optional

from ..token_accounting import TokenAccountingService

logger = logging.getLogger(__name__)

# Lazy-import optional dependencies inside call sites to avoid import-time warnings.


class RankingUtils:
    """Utilities for ranking and filtering retrieval results."""

    def __init__(
        self,
        reranker: Optional[Any] = None,
        chars_per_token: int = 4,
        token_accountant: Optional[TokenAccountingService] = None,
    ):
        """Initialize ranking utilities.

        Args:
            reranker: Optional CrossEncoder for reranking
            chars_per_token: Characters per token estimation
        """
        self.reranker = reranker
        self.chars_per_token = chars_per_token
        self.token_accountant = token_accountant

    def _rerank_results(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank candidates using CrossEncoder."""
        if not self.reranker:
            logger.warning("Reranking not available")
            return candidates

        try:
            # Prepare pairs for reranking
            pairs = [[query, doc["text"]] for doc in candidates]

            # Get reranking scores
            scores = self.reranker.predict(pairs)

            # Update scores
            for doc, score in zip(candidates, scores):
                doc["rerank_score"] = float(score)

            # Sort by reranking score
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

            return candidates

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates

    def _filter_and_rank_chunks(
        self, query: str, chunks: List[Dict[str, Any]], max_chunks: int = 50
    ) -> List[Dict[str, Any]]:
        """Filter and rank chunks based on relevance."""
        if not chunks:
            return []

        filtered_chunks = []
        for chunk in chunks:
            # Additional relevance scoring
            text = chunk.get("text", "")
            score = chunk.get("score", 0)

            # Boost score for exact query term matches
            query_terms = set(query.lower().split())
            chunk_terms = set(text.lower().split())
            term_overlap = (
                len(query_terms.intersection(chunk_terms)) / len(query_terms) if query_terms else 0
            )

            # Combined score
            combined_score = score * (1 + term_overlap)
            chunk["relevance_score"] = combined_score
            filtered_chunks.append(chunk)

        # Sort by score
        filtered_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return filtered_chunks[:max_chunks]

    def _trim_to_token_limit(
        self, chunks: List[Dict[str, Any]], max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Trim chunks to fit within token limit."""
        total_tokens = 0
        trimmed_chunks = []

        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk["text"])
            if total_tokens + chunk_tokens <= max_tokens:
                trimmed_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                break

        return trimmed_chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if self.token_accountant:
            return self.token_accountant.count_tokens(text)
        if not text:
            return 0
        return max(1, len(text) // self.chars_per_token)
