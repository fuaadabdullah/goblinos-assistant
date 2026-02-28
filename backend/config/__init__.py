"""
Environment-Aware Configuration System

Provides centralized configuration management with environment-specific settings,
validation, and computed properties for the Goblin Assistant backend.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings with environment-aware defaults"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    instance_count: int = 1

    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_timeout: int = 5
    # By default memory fallback is disabled to force use of real DBs in production
    allow_memory_fallback: bool = False

    # Authentication
    challenge_ttl: int = 300  # 5 minutes
    debug_auth: bool = False
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # Email Validation
    require_email_validation: bool = True
    allow_disposable_emails: bool = False

    # Database
    database_url: str = ""

    # Logging
    log_level: str = "INFO"

    # RAG Configuration
    enable_enhanced_rag: bool = True  # Enable advanced RAG features (hybrid search, reranking, query expansion)
    rag_chroma_path: str = "data/vector/chroma"  # Path for ChromaDB vector storage
    rag_general_embedding_model: str = "all-MiniLM-L6-v2"  # SentenceTransformers model name
    rag_embedding_backend: Literal["sentence_transformers", "onnx"] = "sentence_transformers"
    rag_onnx_model_dir: str = ""  # Directory containing an exported ONNX model + tokenizer files
    rag_onnx_model_file: str = ""  # Optional override (e.g. "model.int8.onnx"); default auto-detects
    rag_onnx_provider: str = "CPUExecutionProvider"  # onnxruntime execution provider

    # Optional domain-specific embedding models. Keep embeddings in separate collections per content type.
    rag_code_embedding_model: str = "mchochlov/codebert-base-cd-ft"
    rag_legal_embedding_model: str = "sentence-transformers/all-distilroberta-v1"
    rag_scientific_embedding_model: str = "gsarti/scibert-nli"

    rag_query_prefix: str = "query: "  # Used by prompt-aware embedding models
    rag_passage_prefix: str = "passage: "  # Used by prompt-aware embedding models
    rag_instruction_prefix: str = ""  # Optional instruction prefix (e.g. "Represent this for searching:")
    rag_normalize_embeddings: bool = True  # Recommended for cosine similarity retrieval

    @property
    def is_multi_instance(self) -> bool:
        """Check if running in multi-instance mode"""
        return self.instance_count > 1

    @property
    def should_alert_on_fallback(self) -> bool:
        """Determine if fallback mode should trigger alerts"""
        return self.environment == "production" and self.is_multi_instance

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"

# Global settings instance
settings = Settings()


# Environment-specific validation
def validate_environment_config():
    """Validate configuration based on environment"""
    if settings.is_production and not settings.database_url:
        raise ValueError("DATABASE_URL is required in production")

    if (
        settings.is_production
        and settings.allow_memory_fallback
        and settings.is_multi_instance
    ):
        raise ValueError(
            "Memory fallback not allowed in multi-instance production. "
            "Redis is required for distributed challenge storage."
        )

    if settings.environment not in ["development", "staging", "production"]:
        raise ValueError(f"Invalid environment: {settings.environment}")


# Run validation on import
validate_environment_config()
