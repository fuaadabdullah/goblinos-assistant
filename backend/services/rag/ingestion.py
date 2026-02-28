"""
Document ingestion utilities for RAG service.
Handles chunking and document addition to vector stores.
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional

from .hybrid_retriever import HybridRetriever
from ..token_accounting import TokenAccountingService

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    token_accountant: Optional[TokenAccountingService] = None,
) -> List[str]:
    """Chunk text into smaller pieces with overlap."""
    token_accountant = token_accountant or TokenAccountingService()
    return token_accountant.chunk_text(text, chunk_size, overlap)


class DocumentIngestor:
    """Handles document ingestion with chunking and embedding."""

    def __init__(self, service):
        self.service = service

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str = "documents",
        content_type: str = "general",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> bool:
        """Add documents with multiple embedding support."""
        try:
            # Handle fallback mode (no ChromaDB)
            if not self.service.chroma_client:
                embedder = self.service._select_embedder(content_type)
                if hasattr(embedder, 'add_documents'):  # TF-IDF fallback
                    # For TF-IDF, add documents directly to the embedder
                    embedder.add_documents(documents)
                    # Update sparse retriever for hybrid search
                    sparse_docs = [
                        {
                            "id": doc.get("id", i),
                            "text": doc.get("content", doc.get("text", "")),
                        }
                        for i, doc in enumerate(documents)
                    ]
                    self.service.sparse_retriever.add_documents(sparse_docs)
                    # Initialize hybrid retriever
                    self.service.dense_retriever = embedder
                    self.service.hybrid_retriever = HybridRetriever(
                        self.service.dense_retriever, self.service.sparse_retriever
                    )
                    logger.info(
                        f"Added {len(documents)} documents with TF-IDF embeddings to {collection_name}"
                    )
                    return True
                else:
                    logger.error("Unsupported embedder type in fallback mode")
                    return False

            # Full mode with ChromaDB
            collection = self.service.chroma_client.get_or_create_collection(
                name=collection_name
            )

            # Select appropriate embedder
            embedder = self.service._select_embedder(content_type)

            all_chunks = []
            all_embeddings = []
            all_metadatas = []
            all_ids = []

            for doc in documents:
                doc_id = doc.get("id", hashlib.sha256(doc["content"].encode()).hexdigest())
                content = doc["content"]
                metadata = doc.get("metadata", {})

                # Chunk the document
                chunks = chunk_text(
                    content,
                    chunk_size,
                    chunk_overlap,
                    token_accountant=getattr(self.service, "token_accountant", None),
                )

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunk_metadata = {
                        **metadata,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "content_type": content_type,
                        "chunk_text": chunk[:200],
                    }

                    all_chunks.append(chunk)
                    all_metadatas.append(chunk_metadata)
                    all_ids.append(chunk_id)

            # Handle different embedder types
            if hasattr(embedder, 'add_documents'):  # TF-IDF
                # For TF-IDF, we need to fit on all chunks first
                embedder.add_documents(
                    [
                        {"text": chunk, "id": chunk_id}
                        for chunk, chunk_id in zip(all_chunks, all_ids)
                    ]
                )

                # Then encode each chunk
                for chunk in all_chunks:
                    embedding = embedder.encode([chunk]).tolist()[0]
                    all_embeddings.append(embedding)
            else:
                # Standard sentence-transformers encoding
                for chunk in all_chunks:
                    if hasattr(embedder, 'encode_passage'):
                        vec = embedder.encode_passage(chunk)
                    else:
                        vec = embedder.encode(chunk)
                    embedding = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
                    all_embeddings.append(embedding)

            # Add to collection
            collection.add(
                ids=all_ids,
                embeddings=all_embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
            )

            # Update sparse retriever for hybrid search
            sparse_docs = [
                {"id": chunk_id, "text": chunk}
                for chunk_id, chunk in zip(all_ids, all_chunks)
            ]
            self.service.sparse_retriever.add_documents(sparse_docs)
            # Ensure hybrid retriever is ready in full mode (dense retrieval via the service itself).
            try:
                self.service.dense_retriever = self.service
                self.service.hybrid_retriever = HybridRetriever(
                    self.service.dense_retriever, self.service.sparse_retriever
                )
            except Exception:
                # Best-effort; retrieval can still proceed in dense-only mode.
                pass

            logger.info(
                f"Added {len(documents)} documents with {content_type} embeddings to {collection_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
