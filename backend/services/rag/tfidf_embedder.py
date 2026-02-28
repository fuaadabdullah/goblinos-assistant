from __future__ import annotations

"""
TF-IDF Embedder for Enhanced RAG
Fallback embedder using TF-IDF vectorization when sentence-transformers is not available.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False
    logger.warning("TF-IDF dependencies not available")


class TfidfEmbedder:
    """Fallback embedder using TF-IDF vectorization when sentence-transformers is not available."""

    def __init__(self):
        if not TFIDF_AVAILABLE:
            raise ImportError("TF-IDF dependencies not available")

        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit vocabulary size
            stop_words="english",
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,
            max_df=1.0,  # Allow all documents
        )
        self.documents = []
        self.doc_ids = []
        self.tfidf_matrix = None
        self.is_fitted = False

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the TF-IDF index."""
        self.documents = documents
        self.doc_ids = [doc.get("id", i) for i, doc in enumerate(documents)]

        # Extract text content
        texts = [doc.get("text", doc.get("content", "")) for doc in documents]

        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to TF-IDF vectors."""
        if not self.is_fitted:
            raise ValueError("TF-IDF embedder must be fitted with documents first")

        if isinstance(texts, str):
            texts = [texts]

        # Transform query using existing vocabulary
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray()

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve documents using TF-IDF similarity."""
        if not self.is_fitted or self.tfidf_matrix is None:
            return []

        # Encode query
        query_vector = self.encode([query])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                doc = self.documents[idx].copy()
                doc["id"] = self.doc_ids[idx]
                doc["score"] = float(similarities[idx])
                doc["retrieval_type"] = "tfidf"
                results.append(doc)

        return results
