"""
BM25 Retriever for Enhanced RAG
Sparse retrieval using BM25 algorithm with fallback to simple text matching.
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
try:
    import numpy as np
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("BM25 dependencies not available, using text fallback")


class BM25Retriever:
    """Sparse retriever using BM25 algorithm."""

    def __init__(self):
        self.documents = []
        self.bm25 = None
        self.doc_ids = []

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the BM25 index."""
        try:
            self.documents = documents
            self.doc_ids = [doc.get("id", i) for i, doc in enumerate(documents)]

            # Tokenize documents for BM25
            tokenized_docs = [self._tokenize(doc.get("text", "")) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        except NameError:
            # BM25Okapi not available, store documents for basic retrieval
            logger.warning("BM25Okapi not available, using basic text matching")
            self.documents = documents
            self.doc_ids = [doc.get("id", i) for i, doc in enumerate(documents)]
            self.bm25 = None

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve documents using BM25."""
        if not self.bm25:
            # Fallback to simple text matching
            return self._simple_text_retrieval(query, top_k)

        try:
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include relevant results
                    doc = self.documents[idx].copy()
                    doc["id"] = self.doc_ids[idx]
                    doc["score"] = float(scores[idx])
                    doc["retrieval_type"] = "sparse"
                    results.append(doc)

            return results
        except NameError:
            # numpy not available, fallback to simple text matching
            return self._simple_text_retrieval(query, top_k)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Convert to lowercase and split on whitespace and punctuation
        return re.findall(r"\b\w+\b", text.lower())

    def _simple_text_retrieval(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Fallback retrieval using simple text matching when BM25 is not available."""
        query_lower = query.lower()
        results = []

        for i, doc in enumerate(self.documents):
            text = doc.get("text", "").lower()
            # Simple scoring based on word overlap
            query_words = set(self._tokenize(query_lower))
            text_words = set(self._tokenize(text))
            overlap = len(query_words.intersection(text_words))

            if overlap > 0:
                score = overlap / len(query_words)  # Normalized score
                doc_copy = doc.copy()
                doc_copy["id"] = self.doc_ids[i]
                doc_copy["score"] = score
                doc_copy["retrieval_type"] = "simple_text"
                results.append(doc_copy)

        # Sort by score and return top-k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
