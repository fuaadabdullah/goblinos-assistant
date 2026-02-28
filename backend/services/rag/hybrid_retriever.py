"""
Hybrid Retriever
Combines dense retrieval (embeddings) with sparse retrieval (BM25/TF-IDF) for better recall.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Merge dense + sparse results into a single ranked list.

    The `dense_retriever` and `sparse_retriever` are expected to expose:
    - `retrieve(query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]`
    Each result should contain at minimum: `id` and `score` (higher is better).
    """

    def __init__(
        self,
        dense_retriever: Any,
        sparse_retriever: Any,
        *,
        dense_weight: float = 0.6,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = float(dense_weight)

    def retrieve(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        dense = []
        sparse = []

        try:
            dense = self.dense_retriever.retrieve(query, top_k=top_k, filters=filters, **kwargs)
        except Exception as exc:
            logger.warning("Dense retrieval failed: %s", exc)

        try:
            sparse = self.sparse_retriever.retrieve(query, top_k=top_k, filters=filters, **kwargs)
        except Exception as exc:
            logger.warning("Sparse retrieval failed: %s", exc)

        if not dense and not sparse:
            return []

        # Normalize each set to [0, 1] to make weights meaningful.
        def _normalize(scores: List[float]) -> List[float]:
            if not scores:
                return []
            mn = min(scores)
            mx = max(scores)
            if mx - mn <= 1e-9:
                return [1.0 for _ in scores]
            return [(s - mn) / (mx - mn) for s in scores]

        dense_scores = _normalize([float(r.get("score", 0.0)) for r in dense])
        sparse_scores = _normalize([float(r.get("score", 0.0)) for r in sparse])

        merged: Dict[str, Dict[str, Any]] = {}

        for r, s in zip(dense, dense_scores):
            doc_id = str(r.get("id", ""))
            if not doc_id:
                continue
            merged[doc_id] = {**r, "score": self.dense_weight * s, "retrieval_type": "hybrid_dense"}

        for r, s in zip(sparse, sparse_scores):
            doc_id = str(r.get("id", ""))
            if not doc_id:
                continue
            if doc_id in merged:
                merged[doc_id]["score"] = float(merged[doc_id].get("score", 0.0)) + (1.0 - self.dense_weight) * s
                merged[doc_id]["retrieval_type"] = "hybrid"
            else:
                merged[doc_id] = {**r, "score": (1.0 - self.dense_weight) * s, "retrieval_type": "hybrid_sparse"}

        out = list(merged.values())
        out.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        return out[:top_k]


__all__ = ["HybridRetriever"]

