"""
Reranker Service
Cross-encoder reranking for improved retrieval precision
"""

from typing import Optional

import structlog

logger = structlog.get_logger()


class RerankerService:
    """
    Reranker service using Cross-Encoder model.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
    Alternative: cross-encoder/ms-marco-MiniLM-L-12-v2 (higher quality)
    """

    # Model options
    MODELS = {
        "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast, 6 layers
        "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Better, 12 layers
    }

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the reranker service.

        Args:
            model_name: Short name of the model (see MODELS dict)
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.full_model_name = self.MODELS.get(
            model_name, f"cross-encoder/{model_name}"
        )
        self._model = None
        self._device = device

    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model", model=self.full_model_name)
            self._model = CrossEncoder(
                self.full_model_name,
                device=self._device,
            )
            logger.info("Reranker model loaded")
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The query to rank documents against
            documents: List of document texts to rerank
            top_n: Return only top N results (None = all)

        Returns:
            List of (document_index, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)

        # Create index-score pairs and sort
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_n limit
        if top_n is not None:
            indexed_scores = indexed_scores[:top_n]

        return [(idx, float(score)) for idx, score in indexed_scores]

    def rerank_with_docs(
        self,
        query: str,
        documents: list[str],
        top_n: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Rerank documents and return them with scores.

        Args:
            query: The query to rank documents against
            documents: List of document texts to rerank
            top_n: Return only top N results (None = all)

        Returns:
            List of (document_text, score) tuples, sorted by score descending
        """
        reranked = self.rerank(query, documents, top_n)
        return [(documents[idx], score) for idx, score in reranked]


# Singleton instance
_reranker_service: Optional[RerankerService] = None


def get_reranker_service(
    model_name: str = "ms-marco-MiniLM-L-6-v2",
) -> RerankerService:
    """Get or create reranker service singleton."""
    global _reranker_service
    if _reranker_service is None or _reranker_service.model_name != model_name:
        _reranker_service = RerankerService(model_name=model_name)
    return _reranker_service
