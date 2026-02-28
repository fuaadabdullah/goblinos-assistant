"""
Retrieval components for RAG service.
Handles dense and hybrid retrieval with query expansion and reranking.
"""

import logging
from typing import List, Dict, Any, Optional

from .ranking import RankingUtils

logger = logging.getLogger(__name__)


class RetrievalClient:
    """Client for handling retrieval operations."""

    def __init__(self, service):
        self.service = service
        self.ranking_utils = RankingUtils(
            reranker=getattr(service, 'reranker', None),
            chars_per_token=getattr(service, 'chars_per_token', 4),
            token_accountant=getattr(service, 'token_accountant', None),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: str = "documents",
        content_type: str = "general",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Dense retrieval method for hybrid search compatibility."""
        try:
            # Handle fallback mode
            if not hasattr(self.service, 'chroma_client') or not self.service.chroma_client:
                if hasattr(self.service, "_select_embedder"):
                    embedder = self.service._select_embedder(content_type)
                else:
                    embedder = self.service.embedders.get("general")
                if hasattr(embedder, 'retrieve'):
                    return embedder.retrieve(query, top_k=top_k)
                else:
                    return []

            # Full mode with ChromaDB
            # Generate query embedding in the same space as the target collection.
            embedder = (
                self.service._select_embedder(content_type)
                if hasattr(self.service, "_select_embedder")
                else self.service.embedders["general"]
            )
            if hasattr(embedder, "encode_query"):
                vec = embedder.encode_query(query)
            else:
                vec = embedder.encode(query)
            query_embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

            # Search collection
            collection = self.service.chroma_client.get_or_create_collection(name=collection_name)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
            )

            if not results["documents"]:
                return []

            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "text": doc,
                        "metadata": metadata,
                        "score": 1.0 / (1.0 + distance),  # Convert distance to similarity
                        "distance": distance,
                        "retrieval_type": "dense",
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        expand_query: bool = True,
        collection_name: str = "documents",
        content_type: str = "general",
    ) -> Dict[str, Any]:
        """Enhanced retrieval with hybrid search, reranking, and query expansion."""
        try:
            # Query expansion
            if expand_query and hasattr(self.service, 'query_expander'):
                expanded_queries = await self.service.query_expander.expand_query(query)
                logger.info(f"Expanded query to {len(expanded_queries)} variations")
            else:
                expanded_queries = [query]

            all_results = []

            # Retrieve using multiple query variations
            for q in expanded_queries:
                if use_hybrid and hasattr(self.service, 'hybrid_retriever') and self.service.hybrid_retriever:
                    # Hybrid search
                    results = self.service.hybrid_retriever.retrieve(
                        q,
                        top_k=top_k,
                        filters=filters,
                        collection_name=collection_name,
                        content_type=content_type,
                    )
                else:
                    # Dense-only search
                    results = self.retrieve(
                        q,
                        top_k=top_k,
                        filters=filters,
                        collection_name=collection_name,
                        content_type=content_type,
                    )

                all_results.extend(results)

            # Remove duplicates and sort by score
            seen_ids = set()
            unique_results = []
            for result in all_results:
                doc_id = result.get("id")
                if doc_id not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(doc_id)

            unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            candidates = unique_results[: self.service.rerank_top_k]

            # Reranking
            if use_reranking and len(candidates) > 1 and hasattr(self.service, 'reranker') and self.service.reranker:
                reranked_results = self.ranking_utils._rerank_results(query, candidates)
            else:
                reranked_results = candidates

            # Filter and rank final results
            filtered_chunks = self.ranking_utils._filter_and_rank_chunks(query, reranked_results)

            # Trim to token limit
            total_tokens = sum(
                self.ranking_utils._estimate_tokens(chunk["text"]) for chunk in filtered_chunks
            )
            if total_tokens > self.service.max_retriever_tokens:
                filtered_chunks = self.ranking_utils._trim_to_token_limit(
                    filtered_chunks, self.service.max_retriever_tokens
                )

            return {
                "chunks": filtered_chunks,
                "total_tokens": sum(
                    self.ranking_utils._estimate_tokens(chunk["text"]) for chunk in filtered_chunks
                ),
                "filtered_count": len(filtered_chunks),
                "query_expansions": len(expanded_queries),
                "hybrid_search": use_hybrid,
                "reranking": use_reranking,
            }

        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            return {
                "chunks": [],
                "total_tokens": 0,
                "filtered_count": 0,
                "error": str(e),
            }
