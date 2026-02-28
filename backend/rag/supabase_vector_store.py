"""
Supabase Vector Store using pgvector
Cloud-native vector database for RAG with Supabase's PostgreSQL + pgvector extension.
"""

import os
import uuid
import logging
from dataclasses import dataclass
from typing import Any, Optional, List
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document with embedding metadata."""

    id: str
    content: str
    embedding: List[float]
    metadata: dict[str, Any]

    @classmethod
    def create(
        cls,
        content: str,
        embedding: List[float],
        metadata: Optional[dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> "Document":
        return cls(
            id=doc_id or str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )


@dataclass
class SearchResult:
    """Search result from vector store."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any]


# Table name for RAG embeddings (separate from legacy embeddings table)
RAG_EMBEDDINGS_TABLE = "rag_embeddings"


class SupabaseVectorStore:
    """
    Supabase vector store using pgvector for RAG.

    Features:
    - Document storage with embeddings
    - Similarity search using cosine distance
    - Metadata filtering
    - Batch operations
    - Cloud-native (works in serverless)
    """

    def __init__(
        self,
        collection_name: str = "default",
        embedding_dim: int = 384,  # all-MiniLM-L6-v2
        table_name: str = RAG_EMBEDDINGS_TABLE,
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.table_name = table_name

        # Get Supabase credentials
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv(
            "SUPABASE_ANON_KEY"
        )

        if not self.supabase_url or not self.supabase_key:
            logger.warning(
                "Supabase credentials not configured. Vector store will not be available."
            )
            self.client = None
        else:
            self.client = httpx.AsyncClient(
                base_url=f"{self.supabase_url}/rest/v1",
                headers={
                    "apikey": self.supabase_key,
                    "Authorization": f"Bearer {self.supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation",
                },
                timeout=30.0,
            )
            logger.info(
                f"Initialized Supabase vector store with collection: {collection_name}"
            )

    async def health_check(self) -> dict:
        """Check vector store health."""
        if not self.client:
            return {
                "status": "down",
                "error": "Supabase credentials not configured",
            }

        try:
            # Simple query to check connection
            response = await self.client.get(
                f"/{self.table_name}", params={"select": "id", "limit": "1"}
            )

            if response.status_code == 200:
                # Get collection stats
                count_response = await self.client.get(
                    f"/{self.table_name}",
                    params={
                        "select": "id",
                        "collection_name": f"eq.{self.collection_name}",
                    },
                    headers={**self.client.headers, "Prefer": "count=exact"},
                )

                count = int(
                    count_response.headers.get("content-range", "0-0/0").split("/")[-1]
                )

                return {
                    "status": "healthy",
                    "collection": self.collection_name,
                    "document_count": count,
                    "provider": "supabase_pgvector",
                }
            else:
                return {
                    "status": "down",
                    "error": f"HTTP {response.status_code}: {response.text}",
                }
        except Exception as e:
            return {
                "status": "down",
                "error": str(e),
            }

    async def add_document(self, document: Document) -> str:
        """Add a single document to the store."""
        if not self.client:
            raise RuntimeError("Supabase client not initialized")

        data = {
            "id": document.id,
            "collection_name": self.collection_name,
            "content": document.content,
            "embedding": document.embedding,
            "metadata": document.metadata,
        }

        response = await self.client.post(f"/{self.table_name}", json=data)

        if response.status_code not in (200, 201):
            raise RuntimeError(f"Failed to add document: {response.text}")

        logger.debug(
            f"Added document {document.id} to collection {self.collection_name}"
        )
        return document.id

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add multiple documents in batch."""
        if not self.client:
            raise RuntimeError("Supabase client not initialized")

        data = [
            {
                "id": doc.id,
                "collection_name": self.collection_name,
                "content": doc.content,
                "embedding": doc.embedding,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

        response = await self.client.post(f"/{self.table_name}", json=data)

        if response.status_code not in (200, 201):
            raise RuntimeError(f"Failed to add documents: {response.text}")

        logger.info(
            f"Added {len(documents)} documents to collection {self.collection_name}"
        )
        return [doc.id for doc in documents]

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.7,
        metadata_filter: Optional[dict] = None,
    ) -> List[SearchResult]:
        """Search for similar documents using cosine similarity."""
        if not self.client:
            raise RuntimeError("Supabase client not initialized")

        # Use the RPC function for similarity search
        rpc_url = f"{self.supabase_url}/rest/v1/rpc/match_embeddings"

        response = await self.client.post(
            rpc_url.replace(f"{self.supabase_url}/rest/v1", ""),
            json={
                "query_embedding": query_embedding,
                "collection_filter": self.collection_name,
                "match_count": top_k,
                "match_threshold": threshold,
            },
        )

        if response.status_code != 200:
            # Fallback to manual search if RPC not available
            logger.warning(f"RPC search failed, using fallback: {response.text}")
            return await self._fallback_search(query_embedding, top_k, threshold)

        results = response.json()

        # Apply metadata filtering if provided
        if metadata_filter:
            results = [
                r
                for r in results
                if all(
                    r.get("metadata", {}).get(k) == v
                    for k, v in metadata_filter.items()
                )
            ]

        return [
            SearchResult(
                id=r["id"],
                content=r["content"],
                score=r["similarity"],
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    async def _fallback_search(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float,
    ) -> List[SearchResult]:
        """Fallback search using client-side similarity calculation."""
        # Get all documents from collection
        response = await self.client.get(
            f"/{self.table_name}",
            params={
                "collection_name": f"eq.{self.collection_name}",
                "select": "id,content,embedding,metadata",
                "limit": str(top_k * 10),  # Get more to filter
            },
        )

        if response.status_code != 200:
            return []

        docs = response.json()

        # Calculate similarity (cosine)
        import numpy as np

        query_vec = np.array(query_embedding)

        results = []
        for doc in docs:
            if not doc.get("embedding"):
                continue
            doc_vec = np.array(doc["embedding"])
            similarity = float(
                np.dot(query_vec, doc_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            )

            if similarity >= threshold:
                results.append((similarity, doc))

        # Sort by similarity and take top_k
        results.sort(key=lambda x: x[0], reverse=True)
        results = results[:top_k]

        return [
            SearchResult(
                id=doc["id"],
                content=doc["content"],
                score=score,
                metadata=doc.get("metadata", {}),
            )
            for score, doc in results
        ]

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        if not self.client:
            raise RuntimeError("Supabase client not initialized")

        response = await self.client.delete(
            f"/{self.table_name}",
            params={"id": f"eq.{document_id}"},
        )

        return response.status_code == 204

    async def delete_collection(self) -> bool:
        """Delete all documents in the collection."""
        if not self.client:
            raise RuntimeError("Supabase client not initialized")

        response = await self.client.delete(
            f"/{self.table_name}",
            params={"collection_name": f"eq.{self.collection_name}"},
        )

        logger.info(f"Deleted collection {self.collection_name}")
        return response.status_code == 204

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        if not self.client:
            raise RuntimeError("Supabase client not initialized")

        response = await self.client.get(
            f"/{self.table_name}",
            params={
                "id": f"eq.{document_id}",
                "select": "id,content,embedding,metadata",
            },
        )

        if response.status_code != 200 or not response.json():
            return None

        data = response.json()[0]
        return Document(
            id=data["id"],
            content=data["content"],
            embedding=data.get("embedding", []),
            metadata=data.get("metadata", {}),
        )

    async def count(self) -> int:
        """Count documents in the collection."""
        if not self.client:
            return 0

        response = await self.client.get(
            f"/{self.table_name}",
            params={
                "collection_name": f"eq.{self.collection_name}",
                "select": "id",
            },
            headers={**self.client.headers, "Prefer": "count=exact"},
        )

        if response.status_code != 200:
            return 0

        # Parse count from content-range header
        content_range = response.headers.get("content-range", "0-0/0")
        return int(content_range.split("/")[-1])

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()


# Singleton instance
_vector_store: Optional[SupabaseVectorStore] = None


def get_vector_store(collection_name: str = "default") -> SupabaseVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None or _vector_store.collection_name != collection_name:
        _vector_store = SupabaseVectorStore(collection_name=collection_name)
    return _vector_store
