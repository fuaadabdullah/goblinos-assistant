"""
RAG Retriever
Combines vector search with context formatting
"""

from dataclasses import dataclass
from typing import Any, Optional

import structlog

from .vector_store import QdrantVectorStore, SearchResult
from .embeddings import EmbeddingService, get_embedding_service
from .reranker import RerankerService, get_reranker_service

logger = structlog.get_logger()


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""

    query: str
    contexts: list[SearchResult]
    formatted_context: str
    metadata: dict[str, Any]


class RAGRetriever:
    """
    RAG Retriever combining embedding and vector search.

    Features:
    - Query embedding generation
    - Vector similarity search
    - Context formatting for LLM
    - Reranking support
    """

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        top_k: int = 5,
        score_threshold: float = 0.5,
        max_context_length: int = 4000,
    ):
        self.vector_store = vector_store or QdrantVectorStore()
        self.embedding_service = embedding_service or get_embedding_service()
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_context_length = max_context_length

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        include_scores: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant contexts for a query.

        Args:
            query: User query
            top_k: Number of results (overrides default)
            filters: Metadata filters
            include_scores: Include similarity scores in context

        Returns:
            RetrievalResult with contexts and formatted prompt
        """
        k = top_k or self.top_k

        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            limit=k,
            score_threshold=self.score_threshold,
            filters=filters,
        )

        # Format context for LLM
        formatted_context = self._format_context(results, include_scores)

        # Truncate if needed
        if len(formatted_context) > self.max_context_length:
            formatted_context = self._truncate_context(
                results, self.max_context_length, include_scores
            )

        return RetrievalResult(
            query=query,
            contexts=results,
            formatted_context=formatted_context,
            metadata={
                "num_results": len(results),
                "top_score": results[0].score if results else 0.0,
                "filters_applied": filters is not None,
            },
        )

    def _format_context(
        self,
        results: list[SearchResult],
        include_scores: bool = True,
    ) -> str:
        """Format search results as context string."""
        if not results:
            return ""

        parts = []
        for i, result in enumerate(results, 1):
            header = f"[Context {i}]"
            if include_scores:
                header += f" (relevance: {result.score:.2f})"

            parts.append(f"{header}\n{result.content}")

        return "\n\n".join(parts)

    def _truncate_context(
        self,
        results: list[SearchResult],
        max_length: int,
        include_scores: bool = True,
    ) -> str:
        """Truncate context to fit within max length."""
        formatted = ""

        for i, result in enumerate(results, 1):
            header = f"[Context {i}]"
            if include_scores:
                header += f" (relevance: {result.score:.2f})"

            entry = f"{header}\n{result.content}\n\n"

            if len(formatted) + len(entry) > max_length:
                # Add truncated version of last entry if space
                remaining = max_length - len(formatted) - len(header) - 20
                if remaining > 100:
                    truncated_content = result.content[:remaining] + "..."
                    formatted += f"{header}\n{truncated_content}"
                break

            formatted += entry

        return formatted.strip()

    async def retrieve_with_rerank(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_n: int = 3,
        filters: Optional[dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Retrieve with reranking for better precision.

        Uses cross-encoder for reranking (more accurate but slower).
        """
        # First pass: retrieve more candidates
        initial_k = (top_k or self.top_k) * 3
        initial_result = await self.retrieve(
            query=query,
            top_k=initial_k,
            filters=filters,
            include_scores=False,
        )

        if not initial_result.contexts:
            return initial_result

        # Rerank using cached cross-encoder service
        try:
            reranker = get_reranker_service()

            documents = [r.content for r in initial_result.contexts]
            reranked_indices = reranker.rerank(query, documents, top_n=rerank_top_n)

            # Build reranked results
            reranked_results = [
                SearchResult(
                    id=initial_result.contexts[idx].id,
                    content=initial_result.contexts[idx].content,
                    score=score,
                    metadata=initial_result.contexts[idx].metadata,
                )
                for idx, score in reranked_indices
            ]

            formatted_context = self._format_context(reranked_results)

            return RetrievalResult(
                query=query,
                contexts=reranked_results,
                formatted_context=formatted_context,
                metadata={
                    "num_results": len(reranked_results),
                    "reranked": True,
                    "initial_candidates": len(initial_result.contexts),
                },
            )

        except ImportError:
            logger.warning("Cross-encoder not available, returning initial results")
            return initial_result

    def build_rag_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build a RAG prompt for the LLM.

        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt override

        Returns:
            Formatted prompt string
        """
        default_system = """You are a helpful AI assistant. Use the provided context to answer the user's question. If the context doesn't contain relevant information, say so and provide a general response based on your knowledge.

Always cite the context number when using information from it (e.g., "According to Context 1...")."""

        system = system_prompt or default_system

        prompt = f"""{system}

## Retrieved Context
{context}

## User Question
{query}

## Your Response"""

        return prompt
