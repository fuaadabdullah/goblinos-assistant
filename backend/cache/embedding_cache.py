"""
Embedding Cache for document vectors and embeddings.

Provides Redis-backed caching for embeddings and chunked document vectors
with TTL and content hash keys for efficient retrieval.
"""

import hashlib
import json
import pickle
import time
from typing import Optional, Any, Dict, List, Union
import redis
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CachedEmbedding:
    """Cached embedding data structure."""

    content_hash: str
    embedding: List[float]
    model: str
    created_at: float
    ttl_seconds: int
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content_hash": self.content_hash,
            "embedding": self.embedding,
            "model": self.model,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedEmbedding":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CachedPromptResponse:
    """Cached prompt/response pair for RAG optimization."""

    prompt_hash: str
    prompt: str
    response: str
    model: str
    provider: str
    created_at: float
    ttl_seconds: int
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt_hash": self.prompt_hash,
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "provider": self.provider,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedPromptResponse":
        """Create from dictionary."""
        return cls(**data)

    content_hash: str


@dataclass
class CachedDocumentVectors:
    """Cached document vectors with chunking information."""

    content_hash: str
    vectors: List[List[float]]
    chunks: List[str]
    model: str
    chunk_size: int
    overlap: int
    created_at: float
    ttl_seconds: int
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content_hash": self.content_hash,
            "vectors": self.vectors,
            "chunks": self.chunks,
            "model": self.model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedDocumentVectors":
        """Create from dictionary."""
        return cls(**data)


class EmbeddingCache:
    """Redis-backed cache for embeddings and document vectors."""

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        key_prefix: str = "embedding:",
        default_ttl: int = 86400,  # 24 hours
    ):
        """Initialize the embedding cache.

        Args:
            redis_client: Redis client instance (will create if None)
            key_prefix: Prefix for cache keys
            default_ttl: Default TTL in seconds
        """
        self.redis_client = redis_client or self._create_redis_client()
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl

    def _create_redis_client(self) -> Optional[redis.Redis]:
        """Create Redis client from environment variables."""
        try:
            import os

            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                return redis.from_url(redis_url, decode_responses=False)
            else:
                host = os.getenv("REDIS_HOST", "localhost")
                port = int(os.getenv("REDIS_PORT", "6379"))
                db = int(os.getenv("REDIS_DB", "0"))
                password = os.getenv("REDIS_PASSWORD")
                ssl = os.getenv("REDIS_SSL", "false").lower() == "true"

                client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    ssl=ssl,
                    decode_responses=False,  # Keep as bytes for pickle
                )
                client.ping()
                return client
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            return None

    def _get_cache_key(
        self, content_hash: str, model: str, cache_type: str = "embedding"
    ) -> str:
        """Generate cache key.

        Args:
            content_hash: Hash of the content
            model: Model name
            cache_type: Type of cache (embedding, document_vectors)

        Returns:
            Cache key string
        """
        return f"{self.key_prefix}{cache_type}:{model}:{content_hash}"

    def _hash_content(self, content: Union[str, List[str]]) -> str:
        """Generate content hash.

        Args:
            content: Content to hash (string or list of strings)

        Returns:
            SHA256 hash of the content
        """
        if isinstance(content, list):
            content = json.dumps(content, sort_keys=True)

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_embedding(self, content: str, model: str) -> Optional[CachedEmbedding]:
        """Get cached embedding.

        Args:
            content: Content that was embedded
            model: Model used for embedding

        Returns:
            CachedEmbedding or None if not found
        """
        if not self.redis_client:
            return None

        content_hash = self._hash_content(content)
        cache_key = self._get_cache_key(content_hash, model, "embedding")

        try:
            data = self.redis_client.get(cache_key)
            if data:
                cached_data = pickle.loads(data)
                cached_embedding = CachedEmbedding.from_dict(cached_data)

                # Check if expired
                if (
                    time.time() - cached_embedding.created_at
                    > cached_embedding.ttl_seconds
                ):
                    self.redis_client.delete(cache_key)
                    return None

                return cached_embedding
        except Exception as e:
            logger.warning(f"Failed to get embedding from cache: {e}")

        return None

    def set_embedding(
        self,
        content: str,
        embedding: List[float],
        model: str,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Cache an embedding.

        Args:
            content: Content that was embedded
            embedding: Embedding vector
            model: Model used for embedding
            ttl_seconds: TTL in seconds (uses default if None)
            metadata: Additional metadata
        """
        if not self.redis_client:
            return

        content_hash = self._hash_content(content)
        ttl = ttl_seconds or self.default_ttl

        cached_embedding = CachedEmbedding(
            content_hash=content_hash,
            embedding=embedding,
            model=model,
            created_at=time.time(),
            ttl_seconds=ttl,
            metadata=metadata,
        )

        cache_key = self._get_cache_key(content_hash, model, "embedding")

        try:
            data = pickle.dumps(cached_embedding.to_dict())
            self.redis_client.setex(cache_key, ttl, data)
            logger.debug(
                f"Cached embedding for model {model}, hash {content_hash[:8]}..."
            )
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    def get_document_vectors(
        self,
        content: Union[str, List[str]],
        model: str,
        chunk_size: int,
        overlap: int = 0,
    ) -> Optional[CachedDocumentVectors]:
        """Get cached document vectors.

        Args:
            content: Original document content or chunks
            model: Model used for embedding
            chunk_size: Chunk size used
            overlap: Chunk overlap used

        Returns:
            CachedDocumentVectors or None if not found
        """
        if not self.redis_client:
            return None

        content_hash = self._hash_content(content)
        cache_key = f"{self.key_prefix}doc_vectors:{model}:{chunk_size}:{overlap}:{content_hash}"

        try:
            data = self.redis_client.get(cache_key)
            if data:
                cached_data = pickle.loads(data)
                cached_vectors = CachedDocumentVectors.from_dict(cached_data)

                # Check if expired
                if time.time() - cached_vectors.created_at > cached_vectors.ttl_seconds:
                    self.redis_client.delete(cache_key)
                    return None

                return cached_vectors
        except Exception as e:
            logger.warning(f"Failed to get document vectors from cache: {e}")

        return None

    def set_document_vectors(
        self,
        content: Union[str, List[str]],
        chunks: List[str],
        vectors: List[List[float]],
        model: str,
        chunk_size: int,
        overlap: int = 0,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Cache document vectors.

        Args:
            content: Original document content or chunks
            chunks: Text chunks
            vectors: Embedding vectors for each chunk
            model: Model used for embedding
            chunk_size: Chunk size used
            overlap: Chunk overlap used
            ttl_seconds: TTL in seconds (uses default if None)
            metadata: Additional metadata
        """
        if not self.redis_client:
            return

        content_hash = self._hash_content(content)
        ttl = ttl_seconds or self.default_ttl

        cached_vectors = CachedDocumentVectors(
            content_hash=content_hash,
            vectors=vectors,
            chunks=chunks,
            model=model,
            chunk_size=chunk_size,
            overlap=overlap,
            created_at=time.time(),
            ttl_seconds=ttl,
            metadata=metadata,
        )

        cache_key = f"{self.key_prefix}doc_vectors:{model}:{chunk_size}:{overlap}:{content_hash}"

        try:
            data = pickle.dumps(cached_vectors.to_dict())
            self.redis_client.setex(cache_key, ttl, data)
            logger.debug(
                f"Cached document vectors for model {model}, {len(chunks)} chunks"
            )
        except Exception as e:
            logger.warning(f"Failed to cache document vectors: {e}")

    def clear_expired(self):
        """Clear expired cache entries (Redis handles TTL automatically, this is for cleanup)."""
        # Redis handles TTL automatically, but this method could be used for manual cleanup
        # or to implement custom expiration logic if needed
        pass

    def get_prompt_response(
        self, prompt: str, model: str, provider: str
    ) -> Optional[CachedPromptResponse]:
        """Get cached prompt/response pair.

        Args:
            prompt: The prompt text
            model: Model used for generation
            provider: Provider used for generation

        Returns:
            CachedPromptResponse or None if not found
        """
        if not self.redis_client:
            return None

        prompt_hash = self._hash_content(prompt)
        cache_key = self._get_cache_key(
            prompt_hash, f"{model}:{provider}", "prompt_response"
        )

        try:
            data = self.redis_client.get(cache_key)
            if data:
                cached_data = pickle.loads(data)
                cached_response = CachedPromptResponse.from_dict(cached_data)

                # Check if expired
                if (
                    time.time() - cached_response.created_at
                    > cached_response.ttl_seconds
                ):
                    self.redis_client.delete(cache_key)
                    return None

                return cached_response
        except Exception as e:
            logger.warning(f"Failed to get prompt response from cache: {e}")

        return None

    def set_prompt_response(
        self,
        prompt: str,
        response: str,
        model: str,
        provider: str,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Cache a prompt/response pair.

        Args:
            prompt: The prompt text
            response: The response text
            model: Model used for generation
            provider: Provider used for generation
            ttl_seconds: TTL in seconds (uses default if None)
            metadata: Additional metadata
        """
        if not self.redis_client:
            return

        prompt_hash = self._hash_content(prompt)
        ttl = ttl_seconds or self.default_ttl

        cached_response = CachedPromptResponse(
            prompt_hash=prompt_hash,
            prompt=prompt,
            response=response,
            model=model,
            provider=provider,
            created_at=time.time(),
            ttl_seconds=ttl,
            metadata=metadata,
        )

        cache_key = self._get_cache_key(
            prompt_hash, f"{model}:{provider}", "prompt_response"
        )

        try:
            data = pickle.dumps(cached_response.to_dict())
            self.redis_client.setex(cache_key, ttl, data)
            logger.debug(
                f"Cached prompt response for model {model}, provider {provider}, hash {prompt_hash[:8]}..."
            )
        except Exception as e:
            logger.warning(f"Failed to cache prompt response: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        if not self.redis_client:
            return {"status": "disconnected"}

        try:
            info = self.redis_client.info()
            keys = self.redis_client.keys(f"{self.key_prefix}*")

            return {
                "status": "connected",
                "total_keys": len(keys),
                "redis_info": {
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "uptime_days": info.get("uptime_in_days"),
                },
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global cache instance
_cache_instance: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = EmbeddingCache()
    return _cache_instance
