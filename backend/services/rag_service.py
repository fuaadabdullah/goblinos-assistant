"""
RAG (Retrieval-Augmented Generation) Service
Implements a flexible pipeline: fast dense retriever → chunk filter → model with extended context.

Features:
- Dense retrieval with semantic search
- Intelligent chunk filtering and ranking
- Extended context window support (10k tokens)
- Session-based caching for hot-paths
- Fallback to large context models
- Optional enhanced features: hybrid search, reranking, query expansion, multiple embeddings
"""

import os
import asyncio
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .token_accounting import TokenAccountingService

logger = logging.getLogger(__name__)

# Support both execution styles:
# - package import: `import backend.main` (preferred for tests/tools)
# - module import from backend dir: `uvicorn main:app`
try:
    from backend.config import settings as app_settings  # type: ignore
except Exception:  # pragma: no cover
    from config import settings as app_settings  # type: ignore

# Lazy imports for optional dependencies
# Check ChromaDB availability independently
try:
    import chromadb

    CHROMADB_AVAILABLE = True
    logger.info("ChromaDB is available")
except ImportError as e:
    CHROMADB_AVAILABLE = False
    logger.warning(f"ChromaDB not available: {e}")
except Exception as e:
    CHROMADB_AVAILABLE = False
    logger.warning(f"ChromaDB failed to import (likely Python 3.14 compatibility issue): {e}")

SentenceTransformer = None
SENTENCE_TRANSFORMERS_AVAILABLE = None


def _ensure_sentence_transformers() -> bool:
    global SentenceTransformer, SENTENCE_TRANSFORMERS_AVAILABLE
    if SENTENCE_TRANSFORMERS_AVAILABLE is not None:
        return SENTENCE_TRANSFORMERS_AVAILABLE
    try:
        from sentence_transformers import SentenceTransformer as _SentenceTransformer

        SentenceTransformer = _SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except Exception as e:  # catch any import-time failures (C-extensions, ABI mismatches, etc.)
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        logger.warning(
            "sentence-transformers not available or failed to initialize. Enhanced embeddings will be limited. %s",
            e,
        )
    return SENTENCE_TRANSFORMERS_AVAILABLE


from .rag.prompt_aware_embedder import PromptAwareConfig, PromptAwareEmbedder


def _resolve_collection_name(base_collection: str, content_type: str) -> str:
    """Keep embedding spaces separated by default.

    If callers request a non-general content type but keep the default
    `documents` collection, we automatically route to `documents__{content_type}`.
    """

    base = (base_collection or "documents").strip() or "documents"
    ct = (content_type or "general").strip().lower() or "general"
    if ct in ("general", "default"):
        return base
    if "__" in base:
        return base
    return f"{base}__{ct}"


def _is_bge_model(model_name: str) -> bool:
    return str(model_name or "").startswith("BAAI/bge-")


class RAGService:
    """RAG service with dense retriever and extended context support."""

    def __init__(self, chroma_path: str = "data/vector/chroma", enable_enhanced: bool = False):
        """Initialize RAG service with ChromaDB and embedding model."""
        self.chroma_path = chroma_path
        self.enable_enhanced = enable_enhanced

        # Enhanced features (lazy-loaded)
        self._enhanced_service = None

        # Configuration
        self.max_retriever_tokens = 10000  # 10k token window
        self.chunk_size = 512  # tokens per chunk
        self.chunk_overlap = 50  # token overlap between chunks
        self.max_chunks = 20  # Maximum chunks to retrieve
        self.session_cache_ttl = 3600  # 1 hour for session cache

        # Token estimation (rough: ~4 chars per token)
        self.chars_per_token = 4
        self.token_accountant = TokenAccountingService()

        # ChromaDB is optional when enhanced RAG is enabled (EnhancedRAGService supports a TF-IDF fallback).
        self.chroma_client = None
        self.documents_collection = None
        self.sessions_collection = None
        if CHROMADB_AVAILABLE:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.documents_collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={"description": "Document chunks for RAG retrieval"},
            )
            self.sessions_collection = self.chroma_client.get_or_create_collection(
                name="sessions",
                metadata={"description": "Recent session cache for hot-paths"},
            )
        else:
            if not self.enable_enhanced:
                logger.error(
                    "ChromaDB dependencies not available and enhanced RAG is disabled. "
                    "RAG service will not function."
                )

        # Standard embedding model (used only when *not* relying on enhanced service).
        self.embedding_model = None
        self._embedders: dict[str, Any] = {}

        general_model_name = getattr(
            app_settings, "rag_general_embedding_model", "all-MiniLM-L6-v2"
        )
        embedding_backend = getattr(app_settings, "rag_embedding_backend", "sentence_transformers")

        # ONNX backend (optional) for low-memory deployments.
        if str(embedding_backend).lower() == "onnx":
            model_dir = str(getattr(app_settings, "rag_onnx_model_dir", "") or "").strip()
            if model_dir:
                try:
                    from .rag.onnx_embedder import OnnxEmbedder

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
                    if _is_bge_model(str(general_model_name)):
                        self.embedding_model = PromptAwareEmbedder(
                            base,
                            config=PromptAwareConfig(
                                query_prefix=getattr(app_settings, "rag_query_prefix", "query: "),
                                passage_prefix=getattr(
                                    app_settings, "rag_passage_prefix", "passage: "
                                ),
                                instruction_prefix=getattr(
                                    app_settings, "rag_instruction_prefix", ""
                                ),
                                normalize_embeddings=getattr(
                                    app_settings, "rag_normalize_embeddings", True
                                ),
                            ),
                        )
                    else:
                        self.embedding_model = base
                except Exception as exc:
                    logger.warning(
                        "Failed to initialize ONNX embedding backend; falling back to sentence-transformers. %s",
                        exc,
                    )
            else:
                logger.warning(
                    "rag_embedding_backend=onnx but RAG_ONNX_MODEL_DIR is empty; falling back to sentence-transformers."
                )

        # sentence-transformers backend (default)
        if self.embedding_model is None:
            if _ensure_sentence_transformers():
                try:
                    base = SentenceTransformer(str(general_model_name))
                    if _is_bge_model(str(general_model_name)):
                        self.embedding_model = PromptAwareEmbedder(
                            base,
                            config=PromptAwareConfig(
                                query_prefix=getattr(app_settings, "rag_query_prefix", "query: "),
                                passage_prefix=getattr(
                                    app_settings, "rag_passage_prefix", "passage: "
                                ),
                                instruction_prefix=getattr(
                                    app_settings, "rag_instruction_prefix", ""
                                ),
                                normalize_embeddings=getattr(
                                    app_settings, "rag_normalize_embeddings", True
                                ),
                            ),
                        )
                    else:
                        self.embedding_model = base
                except Exception as exc:
                    logger.warning("Failed to initialize embedding model: %s", exc)
                    self.embedding_model = None
            else:
                if not self.enable_enhanced:
                    logger.warning(
                        "sentence-transformers not available. Standard RAG mode is disabled "
                        "(enable enhanced RAG or install sentence-transformers)."
                    )

        if self.embedding_model is not None:
            self._embedders["general"] = self.embedding_model

    def _build_prompt_aware(self, base: Any) -> Any:
        return PromptAwareEmbedder(
            base,
            config=PromptAwareConfig(
                query_prefix=getattr(app_settings, "rag_query_prefix", "query: "),
                passage_prefix=getattr(app_settings, "rag_passage_prefix", "passage: "),
                instruction_prefix=getattr(app_settings, "rag_instruction_prefix", ""),
                normalize_embeddings=getattr(app_settings, "rag_normalize_embeddings", True),
            ),
        )

    def _model_for_content_type(self, content_type: str) -> str | None:
        ct = (content_type or "general").strip().lower()
        if ct in ("general", "default"):
            return str(getattr(app_settings, "rag_general_embedding_model", "all-MiniLM-L6-v2"))
        if ct == "code":
            return str(getattr(app_settings, "rag_code_embedding_model", "") or "")
        if ct == "legal":
            return str(getattr(app_settings, "rag_legal_embedding_model", "") or "")
        if ct in ("scientific", "science", "sci"):
            return str(getattr(app_settings, "rag_scientific_embedding_model", "") or "")
        return ""

    def get_embedder(self, content_type: str = "general") -> Any:
        return self._select_embedder(content_type)

    def _select_embedder(self, content_type: str = "general") -> Any:
        ct = (content_type or "general").strip().lower() or "general"
        if ct in ("default",):
            ct = "general"
        if ct in self._embedders:
            return self._embedders[ct]

        # Only sentence-transformers is supported for non-general domain embedders (for now).
        if not _ensure_sentence_transformers():
            self._embedders[ct] = self._embedders.get("general")
            return self._embedders[ct]

        model_name = (self._model_for_content_type(ct) or "").strip()
        if not model_name:
            self._embedders[ct] = self._embedders.get("general")
            return self._embedders[ct]

        try:
            base = SentenceTransformer(model_name)
            embedder = self._build_prompt_aware(base) if _is_bge_model(model_name) else base
            self._embedders[ct] = embedder
            return embedder
        except Exception as exc:
            logger.warning(
                "Failed to initialize embedder for content_type=%s model=%s; falling back to general. %s",
                ct,
                model_name,
                exc,
            )
            self._embedders[ct] = self._embedders.get("general")
            return self._embedders[ct]

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str = "documents",
        content_type: str = "general",
    ) -> bool:
        """Add documents to the vector database with chunking."""
        # Check if we should use enhanced RAG service (including fallback mode)
        if self.enable_enhanced:
            try:
                enhanced_service = self._get_enhanced_service()
                if enhanced_service:
                    resolved_collection = _resolve_collection_name(collection_name, content_type)
                    return await enhanced_service.add_documents(
                        documents, resolved_collection, content_type
                    )
            except Exception as e:
                logger.warning(f"Enhanced RAG add_documents failed, falling back: {e}")

        # Fallback to standard ChromaDB implementation
        if not CHROMADB_AVAILABLE or self.chroma_client is None:
            logger.error("ChromaDB not available. Cannot add documents.")
            return False

        try:
            resolved_collection = _resolve_collection_name(collection_name, content_type)
            collection = self.chroma_client.get_or_create_collection(name=resolved_collection)

            for doc in documents:
                doc_id = doc.get("id", hashlib.sha256(doc["content"].encode()).hexdigest())
                content = doc["content"]
                metadata = doc.get("metadata", {})

                # Chunk the document
                chunks = self._chunk_text(content, self.chunk_size, self.chunk_overlap)

                # Generate embeddings for chunks
                embeddings = []
                chunk_texts = []
                chunk_metadatas = []

                embedder = self._select_embedder(content_type)
                if not embedder:
                    raise RuntimeError(
                        "Embedding model is not available. Install sentence-transformers, configure ONNX, or enable enhanced RAG."
                    )

                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **metadata,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "content_type": content_type,
                        "chunk_text": chunk[:200],  # Preview for debugging
                    }

                    if hasattr(embedder, "encode_passage"):
                        vec = embedder.encode_passage(chunk)
                    else:
                        vec = embedder.encode(chunk)
                    embeddings.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))
                    chunk_texts.append(chunk)
                    chunk_metadatas.append(chunk_metadata)

                # Add to collection
                collection.add(
                    ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
                    embeddings=embeddings,
                    documents=chunk_texts,
                    metadatas=chunk_metadatas,
                )

            logger.info(f"Added {len(documents)} documents to {resolved_collection}")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Chunk text into smaller pieces with overlap."""
        return self.token_accountant.chunk_text(text, chunk_size, overlap)

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        collection_name: str = "documents",
        content_type: str = "general",
    ) -> Dict[str, Any]:
        """Retrieve relevant context using dense retrieval and filtering."""
        try:
            # Prefer enhanced retrieval when enabled (supports TF-IDF fallback without ChromaDB/torch).
            if self.enable_enhanced:
                try:
                    enhanced_service = self._get_enhanced_service()
                    if enhanced_service:
                        resolved_collection = _resolve_collection_name(
                            collection_name, content_type
                        )
                        return await enhanced_service.retrieve_context(
                            query=query,
                            top_k=top_k,
                            filters=filters,
                            collection_name=resolved_collection,
                            content_type=content_type,
                        )
                except Exception as e:
                    logger.warning(f"Enhanced RAG retrieve_context failed, falling back: {e}")

            if not CHROMADB_AVAILABLE or self.chroma_client is None:
                raise RuntimeError("ChromaDB client not available for standard retrieval")

            resolved_collection = _resolve_collection_name(collection_name, content_type)
            collection = self.chroma_client.get_or_create_collection(name=resolved_collection)

            embedder = self._select_embedder(content_type)
            if not embedder:
                raise RuntimeError(
                    "Embedding model is not available. Install sentence-transformers, configure ONNX, or enable enhanced RAG."
                )

            # Generate query embedding
            if hasattr(embedder, "encode_query"):
                vec = embedder.encode_query(query)
            else:
                vec = embedder.encode(query)
            query_embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

            # Search documents collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, self.max_chunks),  # Get more for filtering
                where=filters,
            )

            if not results["documents"]:
                return {"chunks": [], "total_tokens": 0, "filtered_count": 0}

            # Filter and rank chunks
            filtered_chunks = self._filter_and_rank_chunks(
                query,
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )

            # Calculate total tokens
            total_tokens = sum(self._estimate_tokens(chunk["text"]) for chunk in filtered_chunks)

            # Trim to token limit if needed
            if total_tokens > self.max_retriever_tokens:
                filtered_chunks = self._trim_to_token_limit(
                    filtered_chunks, self.max_retriever_tokens
                )

            return {
                "chunks": filtered_chunks,
                "total_tokens": sum(
                    self._estimate_tokens(chunk["text"]) for chunk in filtered_chunks
                ),
                "filtered_count": len(filtered_chunks),
            }

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "chunks": [],
                "total_tokens": 0,
                "filtered_count": 0,
                "error": str(e),
            }

    def _filter_and_rank_chunks(
        self,
        query: str,
        documents: List[str],
        metadatas: List[Dict],
        distances: List[float],
    ) -> List[Dict[str, Any]]:
        """Filter and rank retrieved chunks based on relevance."""
        chunks = []

        for doc, metadata, distance in zip(documents, metadatas, distances):
            # Calculate relevance score (lower distance = higher relevance)
            relevance_score = 1.0 / (1.0 + distance)  # Convert distance to similarity

            # Additional filtering criteria
            query_terms = set(query.lower().split())
            doc_terms = set(doc.lower().split())
            term_overlap = (
                len(query_terms.intersection(doc_terms)) / len(query_terms) if query_terms else 0
            )

            # Boost score for term overlap
            combined_score = relevance_score * (1 + term_overlap)

            chunks.append(
                {
                    "text": doc,
                    "metadata": metadata,
                    "relevance_score": combined_score,
                    "distance": distance,
                    "term_overlap": term_overlap,
                }
            )

        # Sort by combined score and return top chunks
        chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        return chunks[: self.max_chunks]

    def _trim_to_token_limit(
        self, chunks: List[Dict[str, Any]], max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Trim chunks to fit within token limit, keeping highest scoring chunks."""
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
        return self.token_accountant.count_tokens(text)

    async def cache_session_context(
        self,
        session_id: str,
        context: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Cache context for recent sessions to speed hot-paths."""
        try:
            ttl = ttl_seconds or self.session_cache_ttl

            # Create session cache entry
            cache_entry = {
                "session_id": session_id,
                "context": json.dumps(context),
                "created_at": datetime.now().isoformat(),
                "ttl_seconds": ttl,
                "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat(),
            }

            # Generate embedding for session context (for potential similarity search)
            context_text = f"{context.get('query', '')} {' '.join([c.get('text', '')[:100] for c in context.get('chunks', [])])}"
            embedder = self._select_embedder("general")
            if not embedder:
                raise RuntimeError(
                    "Embedding model is not available. Install sentence-transformers or enable enhanced RAG."
                )
            if hasattr(embedder, "encode_passage"):
                vec = embedder.encode_passage(context_text)
            else:
                vec = embedder.encode(context_text)
            embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

            # Store in sessions collection
            self.sessions_collection.add(
                ids=[session_id],
                embeddings=[embedding],
                documents=[context_text],
                metadatas=[cache_entry],
            )

            logger.info(f"Cached session context for {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache session: {e}")
            return False

    async def get_cached_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached session context if still valid."""
        try:
            results = self.sessions_collection.get(ids=[session_id])

            if not results["metadatas"]:
                return None

            metadata = results["metadatas"][0]

            # Check if expired
            expires_at = datetime.fromisoformat(metadata["expires_at"])
            if datetime.now() > expires_at:
                # Clean up expired cache
                self.sessions_collection.delete(ids=[session_id])
                return None

            # Return cached context
            return json.loads(metadata["context"])

        except Exception as e:
            logger.error(f"Failed to get cached session: {e}")
            return None

    async def generate_rag_prompt(
        self, query: str, context: Dict[str, Any], max_context_tokens: int = 8000
    ) -> str:
        """Generate RAG-enhanced prompt with retrieved context."""
        chunks = context.get("chunks", [])

        if not chunks:
            # No context available, return basic prompt
            return f"Query: {query}\n\nPlease provide a helpful response."

        # Build context from chunks
        context_parts = []
        total_tokens = 0

        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_tokens = self._estimate_tokens(chunk_text)

            if total_tokens + chunk_tokens > max_context_tokens:
                break

            context_parts.append(chunk_text)
            total_tokens += chunk_tokens

        full_context = "\n\n".join(context_parts)

        # Create RAG prompt
        rag_prompt = f"""You are a helpful AI assistant with access to relevant context information.

Context Information:
{full_context}

Query: {query}

Instructions:
- Use the provided context to inform your response
- If the context doesn't contain relevant information, say so clearly
- Be accurate and helpful
- Cite specific information from the context when relevant

Response:"""

        return rag_prompt

    async def rag_pipeline(
        self,
        query: str,
        session_id: Optional[str] = None,
        filters: Optional[Dict] = None,
        collection_name: str = "documents",
        content_type: str = "general",
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve → filter → generate prompt."""
        # Check session cache first (hot-path optimization)
        if session_id:
            cached_context = await self.get_cached_session(session_id)
            if cached_context:
                logger.info(f"Using cached context for session {session_id}")
                prompt = await self.generate_rag_prompt(query, cached_context)
                return {
                    "prompt": prompt,
                    "context": cached_context,
                    "cached": True,
                    "session_id": session_id,
                }

        # Perform retrieval
        context = await self.retrieve_context(
            query,
            top_k=top_k,
            filters=filters,
            collection_name=collection_name,
            content_type=content_type,
        )

        # Cache for future use if session_id provided
        if session_id and context.get("chunks"):
            await self.cache_session_context(
                session_id,
                {
                    "query": query,
                    "chunks": context["chunks"],
                    "timestamp": datetime.now().isoformat(),
                },
            )

        # Generate RAG prompt
        prompt = await self.generate_rag_prompt(query, context)

        return {
            "prompt": prompt,
            "context": context,
            "cached": False,
            "session_id": session_id,
        }

    def _get_enhanced_service(self):
        """Lazy-load enhanced RAG service."""
        if self._enhanced_service is None and self.enable_enhanced:
            try:
                from .enhanced_rag_service import EnhancedRAGService

                self._enhanced_service = EnhancedRAGService(chroma_path=self.chroma_path)
                logger.info("Enhanced RAG service initialized")
            except ImportError as e:
                logger.warning(f"Enhanced RAG service not available: {e}")
                self._enhanced_service = None
        return self._enhanced_service

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
        """Enhanced RAG pipeline with advanced features (hybrid search, reranking, query expansion)."""
        if not self.enable_enhanced:
            logger.info("Enhanced features disabled, falling back to standard RAG pipeline")
            return await self.rag_pipeline(
                query,
                session_id,
                filters,
                collection_name=collection_name,
                content_type=content_type,
                top_k=top_k,
            )

        enhanced_service = self._get_enhanced_service()
        if enhanced_service:
            return await enhanced_service.enhanced_rag_pipeline(
                query=query,
                session_id=session_id,
                filters=filters,
                use_hybrid=use_hybrid,
                use_reranking=use_reranking,
                expand_query=expand_query,
                collection_name=_resolve_collection_name(collection_name, content_type),
                content_type=content_type,
                top_k=top_k,
            )
        else:
            logger.warning("Enhanced service not available, falling back to standard RAG pipeline")
            return await self.rag_pipeline(
                query,
                session_id,
                filters,
                collection_name=collection_name,
                content_type=content_type,
                top_k=top_k,
            )
