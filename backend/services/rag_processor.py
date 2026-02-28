"""
RAG processor service for handling Retrieval-Augmented Generation.
Integrates with the existing RAG service and handles context management.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import time

from .rag_service import RAGService
from config import settings

logger = logging.getLogger(__name__)


def should_use_rag(request_intent: Optional[str], context_length: Optional[str]) -> bool:
    """Determine if RAG should be used for this request."""
    if request_intent == "rag":
        return True

    # Use RAG for long context requests
    if context_length and len(context_length) > 1000:
        return True

    return False


async def process_rag_context(
    request_intent: Optional[str],
    messages: List[Dict[str, str]],
    context: Optional[str],
    max_tokens: int,
) -> Tuple[List[Dict[str, str]], Optional[Dict[str, Any]], int]:
    """Process RAG context and update messages if needed."""
    rag_context = None

    if not should_use_rag(request_intent, context):
        return messages, rag_context, max_tokens

    try:
        rag_service = RAGService(
            enable_enhanced=settings.enable_enhanced_rag,
            chroma_path=settings.rag_chroma_path,
        )
        user_query = messages[-1]["content"] if messages else ""
        session_id = f"session_{hash(user_query + str(context or ''))}"

        if settings.enable_enhanced_rag:
            rag_result = await rag_service.enhanced_rag_pipeline(
                query=user_query,
                session_id=session_id,
                filters={"intent": "rag"} if request_intent == "rag" else None,
            )
        else:
            rag_result = await rag_service.rag_pipeline(
                query=user_query,
                session_id=session_id,
                filters={"intent": "rag"} if request_intent == "rag" else None,
            )

        rag_context = rag_result

        if rag_result.get("context", {}).get("chunks"):
            rag_prompt = rag_result["prompt"]
            messages[-1] = {"role": "user", "content": rag_prompt}
            max_tokens = min(max_tokens * 2, 4096)

            logger.info(
                f"RAG: Retrieved {rag_result['context']['filtered_count']} chunks, "
                f"{rag_result['context']['total_tokens']} tokens"
            )

        return messages, rag_context, max_tokens

    except Exception as e:
        logger.warning(f"RAG processing failed: {e}")
        return messages, rag_context, max_tokens


async def add_documents_to_rag(
    documents: List[Dict[str, Any]], collection_name: str = "documents"
) -> bool:
    """Add documents to the RAG system."""
    try:
        rag_service = RAGService(
            enable_enhanced=settings.enable_enhanced_rag,
            chroma_path=settings.rag_chroma_path,
        )
        return await rag_service.add_documents(documents, collection_name)
    except Exception as e:
        logger.error(f"Failed to add documents to RAG: {e}")
        return False


async def retrieve_rag_context(
    query: str, top_k: int = 10, filters: Optional[Dict] = None
) -> Dict[str, Any]:
    """Retrieve context from RAG system."""
    try:
        rag_service = RAGService(
            enable_enhanced=settings.enable_enhanced_rag,
            chroma_path=settings.rag_chroma_path,
        )
        return await rag_service.retrieve_context(query, top_k, filters)
    except Exception as e:
        logger.error(f"Failed to retrieve RAG context: {e}")
        return {
            "chunks": [],
            "total_tokens": 0,
            "filtered_count": 0,
            "error": str(e),
        }


async def generate_rag_prompt(
    query: str, context: Dict[str, Any], max_context_tokens: int = 8000
) -> str:
    """Generate RAG-enhanced prompt."""
    try:
        rag_service = RAGService(
            enable_enhanced=settings.enable_enhanced_rag,
            chroma_path=settings.rag_chroma_path,
        )
        return await rag_service.generate_rag_prompt(query, context, max_context_tokens)
    except Exception as e:
        logger.error(f"Failed to generate RAG prompt: {e}")
        return f"Query: {query}\n\nPlease provide a helpful response."


class RAGProcessor:
    """
    Service wrapper around the RAG helper functions.
    """

    def should_use_rag(self, request_intent: Optional[str], context_length: Optional[str]) -> bool:
        return should_use_rag(request_intent, context_length)

    async def process_rag_context(
        self,
        request_intent: Optional[str],
        messages: List[Dict[str, str]],
        context: Optional[str],
        max_tokens: int,
    ) -> Tuple[List[Dict[str, str]], Optional[Dict[str, Any]], int]:
        return await process_rag_context(request_intent, messages, context, max_tokens)

    async def add_documents_to_rag(
        self, documents: List[Dict[str, Any]], collection_name: str = "documents"
    ) -> bool:
        return await add_documents_to_rag(documents, collection_name)

    async def retrieve_rag_context(
        self, query: str, top_k: int = 10, filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        return await retrieve_rag_context(query, top_k, filters)

    async def generate_rag_prompt(
        self, query: str, context: Dict[str, Any], max_context_tokens: int = 8000
    ) -> str:
        return await generate_rag_prompt(query, context, max_context_tokens)
