"""
Qdrant Vector Store
Self-hosted vector database for RAG
"""

from dataclasses import dataclass
from typing import Any, Optional
import uuid

import structlog
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from ..config import settings

logger = structlog.get_logger()


@dataclass
class Document:
    """Document with embedding metadata."""
    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
    
    @classmethod
    def create(
        cls,
        content: str,
        embedding: list[float],
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


class QdrantVectorStore:
    """
    Self-hosted Qdrant vector store for RAG.
    
    Features:
    - Document storage with embeddings
    - Hybrid search (dense + sparse)
    - Metadata filtering
    - Batch operations
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_dim: int = 384,  # all-MiniLM-L6-v2
    ):
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.embedding_dim = embedding_dim
        
        # Initialize client
        api_key = settings.qdrant_api_key.get_secret_value()
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=api_key if api_key else None,
        )
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            logger.info("Using existing collection", collection=self.collection_name)
        except (UnexpectedResponse, Exception):
            logger.info("Creating new collection", collection=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE,
                ),
                # Enable indexing for fast search
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000,
                ),
            )
    
    async def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> list[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with embeddings
            batch_size: Batch size for upsert operations
            
        Returns:
            List of document IDs
        """
        ids = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            points = [
                models.PointStruct(
                    id=doc.id,
                    vector=doc.embedding,
                    payload={
                        "content": doc.content,
                        **doc.metadata,
                    }
                )
                for doc in batch
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            
            ids.extend([doc.id for doc in batch])
            logger.debug("Upserted batch", count=len(batch), total=len(ids))
        
        logger.info("Added documents", count=len(ids))
        return ids
    
    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            filters: Metadata filters
            
        Returns:
            List of search results
        """
        # Build filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
            query_filter = models.Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )
        
        return [
            SearchResult(
                id=str(hit.id),
                content=hit.payload.get("content", ""),
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "content"},
            )
            for hit in results
        ]
    
    async def delete(
        self,
        ids: Optional[list[str]] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> int:
        """
        Delete documents by ID or filter.
        
        Args:
            ids: Document IDs to delete
            filters: Metadata filters for deletion
            
        Returns:
            Number of deleted documents
        """
        if ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids),
            )
            return len(ids)
        
        if filters:
            conditions = [
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
                for key, value in filters.items()
            ]
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=conditions)
                ),
            )
            # Can't get exact count without additional query
            return -1
        
        return 0
    
    async def get_collection_info(self) -> dict[str, Any]:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
            "optimizer_status": info.optimizer_status.status.value,
        }
    
    def close(self) -> None:
        """Close the client connection."""
        self.client.close()
