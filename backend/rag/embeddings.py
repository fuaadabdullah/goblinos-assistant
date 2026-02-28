"""
Embedding Service
Generate embeddings for documents and queries.

Primary backend: sentence-transformers (local, no cost).
Fallback backend: OpenAI text-embedding-3-small (API-based, production-safe).
"""

import os

import structlog

logger = structlog.get_logger()


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
    except Exception as exc:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        logger.warning("sentence-transformers not available: %s", exc)
    return SENTENCE_TRANSFORMERS_AVAILABLE


class EmbeddingService:
    """
    Embedding service with automatic backend selection.

    Preferred: sentence-transformers (local, no API cost).
    Fallback:  OpenAI text-embedding-3-small (requires OPENAI_API_KEY).

    Default local model: paraphrase-multilingual-MiniLM-L12-v2 (384 dims, multilingual)
    """

    # Local model options with dimensions
    MODELS = {
        "all-MiniLM-L6-v2": 384,  # Fast, good for most use cases
        "all-mpnet-base-v2": 768,  # Higher quality
        "paraphrase-multilingual-MiniLM-L12-v2": 384,  # Multilingual
    }

    # OpenAI model dimensions
    OPENAI_MODEL = "text-embedding-3-small"
    OPENAI_DIM = 1536

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str | None = None,
    ):
        self.model_name = model_name

        if _ensure_sentence_transformers():
            logger.info("Loading local embedding model", model=model_name)
            self.model = SentenceTransformer(
                model_name,
                device=device,
            )
            self._backend = "sentence_transformers"
            self.embedding_dim = self.MODELS.get(model_name, 384)
            logger.info("Embedding model loaded (local)", dim=self.embedding_dim)
        else:
            # Fallback: OpenAI API embeddings
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            if not openai_key:
                raise RuntimeError(
                    "sentence-transformers not available and OPENAI_API_KEY is not set. "
                    "Cannot initialise embedding service."
                )
            try:
                from openai import OpenAI  # noqa: PLC0415

                self._openai_client = OpenAI(api_key=openai_key)
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers not available and openai package is missing."
                ) from exc
            self._backend = "openai"
            self.embedding_dim = self.OPENAI_DIM
            self.model = None
            logger.info(
                "Embedding service initialised with OpenAI fallback",
                model=self.OPENAI_MODEL,
                dim=self.embedding_dim,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if self._backend == "sentence_transformers":
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding.tolist()
        else:
            response = self._openai_client.embeddings.create(
                model=self.OPENAI_MODEL,
                input=text,
            )
            return response.data[0].embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if self._backend == "sentence_transformers":
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            return embeddings.tolist()
        else:
            # OpenAI API supports up to 2048 inputs per request
            results: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                response = self._openai_client.embeddings.create(
                    model=self.OPENAI_MODEL,
                    input=chunk,
                )
                # API returns items sorted by index
                results.extend(
                    [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                )
            return results

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim


# Singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None or _embedding_service.model_name != model_name:
        _embedding_service = EmbeddingService(model_name=model_name)
    return _embedding_service
