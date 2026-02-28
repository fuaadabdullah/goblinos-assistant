"""
Enhanced RAG (Retrieval-Augmented Generation) Service
Implements advanced RAG with multiple embedding models, hybrid search, reranking, and query expansion.

Features:
- Multiple embedding models for different content types (general, code, legal, scientific, etc.)
- Hybrid search combining dense retrieval (ChromaDB) with sparse retrieval (BM25)
- CrossEncoder reranking for improved result quality
- Query expansion with synonyms, related terms, and acronym expansion
- Extended context window support (10k tokens)
- Session-based caching for hot-paths
"""

import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

from config import settings as app_settings

# Lazy imports for optional dependencies
try:
    import chromadb

    CHROMADB_AVAILABLE = True
except (ImportError, Exception) as e:
    chromadb = None
    CHROMADB_AVAILABLE = False
    logger.warning(f"ChromaDB not available: {e}")

SentenceTransformer = None
CrossEncoder = None
np = None

if CHROMADB_AVAILABLE:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer, CrossEncoder

        ENHANCED_RAG_AVAILABLE = True
        logger.info("Full enhanced RAG available (sentence-transformers + ChromaDB)")
    except ImportError:
        try:
            import numpy as np

            ENHANCED_RAG_AVAILABLE = "fallback"
            logger.info("Using TF-IDF fallback for enhanced RAG (torch not available)")
        except ImportError:
            ENHANCED_RAG_AVAILABLE = False
            logger.warning(
                "Enhanced RAG dependencies not available. Enhanced features will be disabled."
            )
else:
    try:
        import numpy as np

        ENHANCED_RAG_AVAILABLE = "fallback"
        logger.info("Using TF-IDF fallback for enhanced RAG (ChromaDB not available)")
    except ImportError:
        ENHANCED_RAG_AVAILABLE = False
        logger.warning(
            "Enhanced RAG dependencies not available. Enhanced features will be disabled."
        )

# Import modular components
from .hybrid_retriever import HybridRetriever
from .bm25_retriever import BM25Retriever
from .tfidf_embedder import TfidfEmbedder
from .query_expansion import QueryExpansion
from .ingestion import DocumentIngestor, chunk_text
from .retrieval import RetrievalClient
from .session_cache import SessionCache
from .pipeline import RAGPipeline
from .ranking import RankingUtils
from ..token_accounting import TokenAccountingService
from .prompt_aware_embedder import PromptAwareConfig, PromptAwareEmbedder


def _resolve_collection_name(base_collection: str, content_type: str) -> str:
    base = (base_collection or "documents").strip() or "documents"
    ct = (content_type or "general").strip().lower() or "general"
    if ct in ("general", "default"):
        return base
    if "__" in base:
        return base
    return f"{base}__{ct}"


def _is_bge_model(model_name: str) -> bool:
    return str(model_name or "").startswith("BAAI/bge-")


class EnhancedRAGService:
    """Enhanced RAG service with multiple embedding models, hybrid search, and reranking."""

    def __init__(self, chroma_path: str = "data/vector/chroma"):
        """Initialize enhanced RAG service."""
        if not ENHANCED_RAG_AVAILABLE:
            logger.error(
                "Enhanced RAG dependencies not available. Cannot initialize enhanced service."
            )
            self.chroma_client = None
            self.embedders = {}
            self.embedding_model = None
            self.reranker = None
            self.query_expander = None
            self.dense_retriever = None
            self.sparse_retriever = None
            self.hybrid_retriever = None
            self.documents_collection = None
            self.sessions_collection = None
            return

        # Configuration (must be set before wiring dependent components)
        self.max_retriever_tokens = 10000  # 10k token window
        self.chunk_size = 512  # tokens per chunk
        self.chunk_overlap = 50  # token overlap
        self.max_chunks = 20  # Maximum chunks to retrieve initially
        self.rerank_top_k = 10  # Number of docs to rerank
        self.session_cache_ttl = 3600  # 1 hour

        # Check if we're using fallback mode or if ChromaDB is not available
        if ENHANCED_RAG_AVAILABLE == "fallback" or not CHROMADB_AVAILABLE:
            # Use TF-IDF fallback embedder
            self.embedders = {
                "general": TfidfEmbedder(),
                "code": TfidfEmbedder(),
                "multilingual": TfidfEmbedder(),
                "medical": TfidfEmbedder(),
                "legal": TfidfEmbedder(),
                "scientific": TfidfEmbedder(),
            }
            self.embedding_model = self.embedders["general"]
            self.reranker = None  # No reranking in fallback mode
            self.chroma_client = None  # No ChromaDB in fallback mode
            self.documents_collection = None
            self.sessions_collection = None
            self._embedder_model_names = {}
            logger.info("Using TF-IDF fallback for enhanced RAG features (ChromaDB not available)")
        else:
            # Full enhanced RAG with sentence-transformers and ChromaDB
            self.chroma_path = chroma_path
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)

            # Multiple embedding models for different content types
            general_model_name = getattr(
                app_settings, "rag_general_embedding_model", "all-MiniLM-L6-v2"
            )
            embedding_backend = str(
                getattr(app_settings, "rag_embedding_backend", "sentence_transformers")
                or "sentence_transformers"
            ).lower()

            general_embedder = None
            if embedding_backend == "onnx":
                model_dir = str(getattr(app_settings, "rag_onnx_model_dir", "") or "").strip()
                if model_dir:
                    try:
                        from .onnx_embedder import OnnxEmbedder

                        base = OnnxEmbedder(
                            model_dir=model_dir,
                            model_file=str(
                                getattr(app_settings, "rag_onnx_model_file", "") or ""
                            ).strip()
                            or None,
                            provider=str(
                                getattr(app_settings, "rag_onnx_provider", "CPUExecutionProvider")
                                or "CPUExecutionProvider"
                            ),
                            normalize_embeddings=bool(
                                getattr(app_settings, "rag_normalize_embeddings", True)
                            ),
                        )
                        general_embedder = base
                    except Exception as exc:
                        logger.warning(
                            "Failed to initialize ONNX embedder for enhanced RAG; falling back to sentence-transformers. %s",
                            exc,
                        )

            if general_embedder is None:
                general_embedder = SentenceTransformer(str(general_model_name))

            if _is_bge_model(str(general_model_name)):
                general_embedder = PromptAwareEmbedder(
                    general_embedder,
                    config=PromptAwareConfig(
                        query_prefix=getattr(app_settings, "rag_query_prefix", "query: "),
                        passage_prefix=getattr(app_settings, "rag_passage_prefix", "passage: "),
                        instruction_prefix=getattr(app_settings, "rag_instruction_prefix", ""),
                        normalize_embeddings=getattr(
                            app_settings, "rag_normalize_embeddings", True
                        ),
                    ),
                )

            # Lazy-loaded per content type to avoid unnecessary downloads / RAM at startup.
            self._embedder_model_names = {
                "general": str(general_model_name),
                "code": str(getattr(app_settings, "rag_code_embedding_model", "") or ""),
                "legal": str(getattr(app_settings, "rag_legal_embedding_model", "") or ""),
                "scientific": str(
                    getattr(app_settings, "rag_scientific_embedding_model", "") or ""
                ),
                # Keep existing optional types (can be wired later via config).
                "multilingual": str(
                    getattr(app_settings, "rag_multilingual_embedding_model", "") or ""
                ),
                "medical": str(getattr(app_settings, "rag_medical_embedding_model", "") or ""),
            }

            self.embedders = {"general": general_embedder}

            # Default to general embedder
            self.embedding_model = self.embedders["general"]

            # Reranking model
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            # Create/get collections
            self.documents_collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"description": "Enhanced document chunks with multiple embeddings"},
            )

            self.sessions_collection = self.chroma_client.get_or_create_collection(
                name="sessions",
                metadata={"description": "Enhanced session cache with multi-modal support"},
            )

        # Query expansion (works in both modes)
        self.query_expander = QueryExpansion()

        # Hybrid search components
        self.dense_retriever = None  # Will be initialized with documents
        self.sparse_retriever = BM25Retriever()
        self.hybrid_retriever = None

        # Document ingestor
        self.ingestor = DocumentIngestor(self)

        # Retrieval client
        self.retrieval_client = RetrievalClient(self)

        # Session cache
        self.session_cache = SessionCache(
            self.sessions_collection, self.embedders, self.session_cache_ttl
        )

        # RAG pipeline
        self.pipeline = RAGPipeline(self)

        # Token estimation
        self.chars_per_token = 4
        self.token_accountant = TokenAccountingService()

        # Ranking utilities
        self.ranking_utils = RankingUtils(
            reranker=getattr(self, "reranker", None),
            chars_per_token=self.chars_per_token,
            token_accountant=self.token_accountant,
        )

    def _select_embedder(self, content_type: str = "general"):
        """Select appropriate embedding model based on content type."""
        ct = (content_type or "general").strip().lower() or "general"
        if ct in ("default",):
            ct = "general"
        if ct in self.embedders:
            return self.embedders[ct]

        model_name = (self._embedder_model_names.get(ct) or "").strip()
        if not model_name:
            # No explicit model configured for this type; fall back to general.
            self.embedders[ct] = self.embedders["general"]
            return self.embedders[ct]

        try:
            embedder = SentenceTransformer(model_name)
            if _is_bge_model(model_name):
                embedder = PromptAwareEmbedder(
                    embedder,
                    config=PromptAwareConfig(
                        query_prefix=getattr(app_settings, "rag_query_prefix", "query: "),
                        passage_prefix=getattr(app_settings, "rag_passage_prefix", "passage: "),
                        instruction_prefix=getattr(app_settings, "rag_instruction_prefix", ""),
                        normalize_embeddings=getattr(
                            app_settings, "rag_normalize_embeddings", True
                        ),
                    ),
                )
            self.embedders[ct] = embedder
            return embedder
        except Exception as exc:
            logger.warning(
                "Failed to initialize embedder for content_type=%s model=%s; falling back to general. %s",
                ct,
                model_name,
                exc,
            )
            self.embedders[ct] = self.embedders["general"]
            return self.embedders[ct]

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str = "documents",
        content_type: str = "general",
    ) -> bool:
        """Add documents with multiple embedding support."""
        collection_name = _resolve_collection_name(collection_name, content_type)
        return await self.ingestor.add_documents(
            documents, collection_name, content_type, self.chunk_size, self.chunk_overlap
        )

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Chunk text into smaller pieces with overlap."""
        return self.token_accountant.chunk_text(text, chunk_size, overlap)

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Dense retrieval method for hybrid search compatibility."""
        return self.retrieval_client.retrieve(query, top_k, **kwargs)

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
        return await self.retrieval_client.retrieve_context(
            query,
            top_k,
            filters,
            use_hybrid,
            use_reranking,
            expand_query,
            collection_name=_resolve_collection_name(collection_name, content_type),
            content_type=content_type,
        )

    async def cache_session_context(
        self,
        session_id: str,
        context: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Cache context for recent sessions."""
        return await self.session_cache.cache_session_context(session_id, context, ttl_seconds)

    async def get_cached_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached session context."""
        return await self.session_cache.get_cached_session(session_id)

    async def generate_rag_prompt(
        self, query: str, context: Dict[str, Any], max_context_tokens: int = 8000
    ) -> str:
        """Generate enhanced RAG prompt with retrieved context."""
        return await self.pipeline.generate_rag_prompt(query, context, max_context_tokens)

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
        return await self.pipeline.enhanced_rag_pipeline(
            query,
            session_id,
            filters,
            use_hybrid,
            use_reranking,
            expand_query,
            collection_name=_resolve_collection_name(collection_name, content_type),
            content_type=content_type,
            top_k=top_k,
        )
