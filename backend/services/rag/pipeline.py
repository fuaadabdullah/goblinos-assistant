"""
Pipeline orchestration for RAG service.
Handles prompt generation and end-to-end pipeline.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates the complete RAG pipeline."""

    def __init__(self, service):
        self.service = service

    async def generate_rag_prompt(
        self, query: str, context: Dict[str, Any], max_context_tokens: int = 8000
    ) -> str:
        """Generate enhanced RAG prompt with retrieved context."""
        chunks = context.get("chunks", [])

        if not chunks:
            return f"Query: {query}\n\nPlease provide a helpful response."

        # Build context with metadata
        context_parts = []
        total_tokens = 0

        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_tokens = self._estimate_tokens(chunk_text)

            if total_tokens + chunk_tokens > max_context_tokens:
                break

            # Add metadata info
            metadata = chunk.get("metadata", {})
            source_info = ""
            if metadata.get("source"):
                source_info = f" [Source: {metadata['source']}]"
            if metadata.get("content_type"):
                source_info += f" [Type: {metadata['content_type']}]"

            context_parts.append(f"{chunk_text}{source_info}")
            total_tokens += chunk_tokens

        full_context = "\n\n".join(context_parts)

        # Enhanced RAG prompt
        rag_prompt = f"""You are an advanced AI assistant with access to comprehensive context information retrieved using state-of-the-art RAG techniques.

Context Information (Retrieved via Hybrid Search + Reranking):
{full_context}

Query: {query}

Instructions:
- Use the provided context to inform your response with specific details
- If the context doesn't contain relevant information, clearly state this
- Be accurate, helpful, and cite specific information from the context
- Consider the source types and content domains when formulating your response
- If multiple perspectives exist in the context, acknowledge them

Response:"""

        return rag_prompt

    async def enhanced_rag_pipeline(
        self,
        query: str,
        session_id: Optional[str] = None,
        filters: Optional[Dict] = None,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        expand_query: bool = True,
        collection_name: str = "documents",
        content_type: str = "general",
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Complete enhanced RAG pipeline with all advanced features."""
        # Check session cache first
        if session_id:
            try:
                cached_context = await self.service.get_cached_session(session_id)
                if cached_context:
                    logger.info(f"Using cached context for session {session_id}")
                    prompt = await self.generate_rag_prompt(query, cached_context)
                    return {
                        "prompt": prompt,
                        "context": cached_context,
                        "cached": True,
                        "session_id": session_id,
                        "features_used": ["session_cache"],
                    }
            except Exception as e:
                logger.warning(f"Session cache not available: {e}")

        # Perform enhanced retrieval
        context = await self.service.retrieve_context(
            query,
            top_k=top_k,
            filters=filters,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
            expand_query=expand_query,
            collection_name=collection_name,
            content_type=content_type,
        )

        # Cache for future use
        if session_id and context.get("chunks"):
            try:
                await self.service.cache_session_context(
                    session_id,
                    {
                        "query": query,
                        "chunks": context["chunks"],
                        "timestamp": datetime.now().isoformat(),
                        "features_used": {
                            "hybrid_search": use_hybrid,
                            "reranking": use_reranking,
                            "query_expansion": expand_query,
                        },
                    },
                )
            except Exception as e:
                logger.warning(f"Session cache not available: {e}")

        # Generate enhanced RAG prompt
        prompt = await self.generate_rag_prompt(query, context)

        features_used = []
        if use_hybrid:
            features_used.append("hybrid_search")
        if use_reranking:
            features_used.append("reranking")
        if expand_query:
            features_used.append("query_expansion")

        return {
            "prompt": prompt,
            "context": context,
            "cached": False,
            "session_id": session_id,
            "features_used": features_used,
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        token_accountant = getattr(self.service, "token_accountant", None)
        if token_accountant:
            return token_accountant.count_tokens(text)
        if not text:
            return 0
        return max(1, len(text) // self.service.chars_per_token)
