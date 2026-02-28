"""
Session cache management for RAG service.
Handles caching of session contexts with TTL.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SessionCache:
    """Manages session-based caching for RAG contexts."""

    def __init__(self, sessions_collection, embedders, default_ttl: int = 3600):
        self.sessions_collection = sessions_collection
        self.embedders = embedders
        self.default_ttl = default_ttl

    async def cache_session_context(
        self,
        session_id: str,
        context: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Cache context for recent sessions."""
        try:
            ttl = ttl_seconds or self.default_ttl

            cache_entry = {
                "session_id": session_id,
                "context": json.dumps(context),
                "created_at": datetime.now().isoformat(),
                "ttl_seconds": ttl,
                "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat(),
            }

            # Create embedding for session context
            context_text = f"{context.get('query', '')} {' '.join([c.get('text', '')[:100] for c in context.get('chunks', [])])}"
            embedder = self.embedders["general"]
            if hasattr(embedder, "encode_passage"):
                vec = embedder.encode_passage(context_text)
            else:
                vec = embedder.encode(context_text)
            embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

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
        """Retrieve cached session context."""
        try:
            results = self.sessions_collection.get(ids=[session_id])

            if not results["metadatas"]:
                return None

            metadata = results["metadatas"][0]

            # Check expiration
            expires_at = datetime.fromisoformat(metadata["expires_at"])
            if datetime.now() > expires_at:
                self.sessions_collection.delete(ids=[session_id])
                return None

            return json.loads(metadata["context"])

        except Exception as e:
            logger.error(f"Failed to get cached session: {e}")
            return None
