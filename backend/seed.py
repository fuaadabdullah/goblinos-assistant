# apps/goblin-assistant-root/backend/seed.py
import logging
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# Load environment variables from .env file
load_dotenv()

# Import models
from .models import Provider, Model, SearchCollection, SearchDocument, Task

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _provider_seed_data() -> list[dict[str, Any]]:
    siliconeflow_api_key = _env("SILICONEFLOW_API_KEY") or _env("SILICONFLOW_API_KEY")
    azure_api_key = (
        _env("AZURE_API_KEY") or _env("AZURE_OPENAI_API_KEY") or _env("AZURE_OPENAI_KEY")
    )
    azure_base_url = _env("AZURE_OPENAI_ENDPOINT", "https://{resource}.openai.azure.com")
    aliyun_api_key = _env("ALIYUN_MODEL_SERVER_KEY") or _env("ALIYUN_API_KEY")
    aliyun_base_url = _env(
        "ALIYUN_MODEL_SERVER_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    providers: list[dict[str, Any]] = [
        {
            "name": "openai",
            "api_key": _env("OPENAI_API_KEY"),
            "base_url": "",
            "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "enabled": True,
        },
        {
            "name": "anthropic",
            "api_key": _env("ANTHROPIC_API_KEY"),
            "base_url": "",
            "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "enabled": True,
        },
        {
            "name": "gemini",
            "api_key": _env("GEMINI_API_KEY"),
            "base_url": "",
            "models": ["gemini-pro", "gemini-pro-vision"],
            "enabled": True,
        },
        {
            "name": "groq",
            "api_key": _env("GROQ_API_KEY"),
            "base_url": "",
            "models": ["llama2-70b-4096", "mixtral-8x7b-32768"],
            "enabled": True,
        },
        {
            "name": "deepseek",
            "api_key": _env("DEEPSEEK_API_KEY"),
            "base_url": "",
            "models": ["deepseek-chat", "deepseek-coder"],
            "enabled": True,
        },
        {
            "name": "azure_openai",
            "api_key": azure_api_key,
            "base_url": azure_base_url,
            "models": ["gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-3.5-turbo"],
            "enabled": True,
        },
        {
            "name": "aliyun",
            "api_key": aliyun_api_key,
            "base_url": aliyun_base_url,
            "models": [
                "qwen-plus",
                "qwen-turbo",
                "qwen-max",
                "qwen2.5-coder-32b-instruct",
            ],
            "enabled": True,
        },
        {
            "name": "siliconeflow",
            "api_key": siliconeflow_api_key,
            "base_url": "",
            "models": ["Qwen2-72B-Instruct"],
            "enabled": True,
        },
        {
            "name": "moonshot",
            "api_key": _env("MOONSHOT_API_KEY"),
            "base_url": "",
            "models": ["moonshot-v1-8k", "moonshot-v1-32k"],
            "enabled": True,
        },
        {
            "name": "fireworks",
            "api_key": _env("FIREWORKS_API_KEY"),
            "base_url": "",
            "models": ["accounts/fireworks/models/llama-v2-7b-chat"],
            "enabled": True,
        },
        {
            "name": "elevenlabs",
            "api_key": _env("ELEVENLABS_API_KEY"),
            "base_url": "",
            "models": ["eleven_monolingual_v1"],
            "enabled": True,
        },
        {
            "name": "datadog",
            "api_key": _env("DATADOG_API_KEY"),
            "base_url": "",
            "models": [],
            "enabled": True,
        },
        {
            "name": "netlify",
            "api_key": _env("NETLIFY_API_KEY"),
            "base_url": "",
            "models": [],
            "enabled": True,
        },
    ]

    ollama_gcp_url = _env("OLLAMA_GCP_BASE_URL") or _env("OLLAMA_GCP_URL")
    llamacpp_gcp_url = _env("LLAMACPP_GCP_BASE_URL") or _env("LLAMACPP_GCP_URL")
    gcp_api_key = _env("GCP_LLM_API_KEY") or _env("LOCAL_LLM_API_KEY")
    if ollama_gcp_url:
        providers.append(
            {
                "name": "ollama_gcp",
                "api_key": gcp_api_key,
                "base_url": ollama_gcp_url,
                "models": [
                    "phi3:3.8b",
                    "gemma:2b",
                    "qwen2.5:3b",
                    "deepseek-coder:1.3b",
                    "mistral:7b",
                ],
                "enabled": True,
            }
        )
    if llamacpp_gcp_url:
        providers.append(
            {
                "name": "llamacpp_gcp",
                "api_key": gcp_api_key,
                "base_url": llamacpp_gcp_url,
                "models": [
                    "phi-3-mini-4k-instruct-q4",
                    "llama-2-7b-chat-q4_k_m",
                    "mistral-7b-instruct-v0.2-q4_k_m",
                ],
                "enabled": True,
            }
        )

    # Add TinyLlama as a local provider
    providers.append(
        {
            "name": "tinylama",
            "api_key": "",
            "base_url": "",
            "models": ["tinylama-1.1b-chat"],
            "enabled": True,
        }
    )

    return providers


def _upsert_provider(db: Session, provider_data: dict[str, Any]) -> Provider:
    target_name = cast(str, provider_data["name"])
    provider = db.query(Provider).filter_by(name=target_name).first()
    # Legacy rename: "siliconflow" -> "siliconeflow"
    if not provider and target_name == "siliconeflow":
        provider = db.query(Provider).filter_by(name="siliconflow").first()
        if provider:
            provider_any = cast(Any, provider)
            provider_any.name = "siliconeflow"  # type: ignore[assignment]
            provider = provider_any
    if provider:
        provider.api_key = provider_data.get("api_key") or provider.api_key
        provider.base_url = provider_data.get("base_url", provider.base_url)
        provider.models = provider_data.get("models", provider.models)
        provider.enabled = provider_data.get("enabled", provider.enabled)
        provider.display_name = provider_data["name"].capitalize()
        provider.is_active = provider.enabled
        return provider

    provider_data = {
        **provider_data,
        "display_name": provider_data["name"].capitalize(),
        "is_active": provider_data.get("enabled", True),
    }
    provider = Provider(**provider_data)
    db.add(provider)
    return provider


def seed_database(db: Session) -> None:
    """Seed the database with initial data."""

    # Seed initial data if tables are empty
    providers_data = _provider_seed_data()
    if db.query(Provider).count() == 0:
        logger.info("Seeding initial providers...")
        for provider_data in providers_data:
            _upsert_provider(db, provider_data)
        db.commit()
        logger.info("Providers seeded successfully.")
    else:
        logger.info("Updating existing providers...")
        for provider_data in providers_data:
            provider = _upsert_provider(db, provider_data)
            if provider.api_key and provider_data.get("api_key"):
                logger.info("Updated API key for %s", provider.name)
        db.commit()
        logger.info("Providers updated successfully.")

    if db.query(Model).count() == 0:
        logger.info("Seeding initial models...")
        # Seed models
        models_data = [
            # OpenAI models
            {
                "name": "gpt-4",
                "provider": "openai",
                "model_id": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            {
                "name": "gpt-4-turbo",
                "provider": "openai",
                "model_id": "gpt-4-turbo-preview",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            {
                "name": "gpt-3.5-turbo",
                "provider": "openai",
                "model_id": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            # Anthropic models
            {
                "name": "claude-3-opus",
                "provider": "anthropic",
                "model_id": "claude-3-opus-20240229",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            {
                "name": "claude-3-sonnet",
                "provider": "anthropic",
                "model_id": "claude-3-sonnet-20240229",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            {
                "name": "claude-3-haiku",
                "provider": "anthropic",
                "model_id": "claude-3-haiku-20240307",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            # Gemini models
            {
                "name": "gemini-pro",
                "provider": "gemini",
                "model_id": "gemini-pro",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            {
                "name": "gemini-pro-vision",
                "provider": "gemini",
                "model_id": "gemini-pro-vision",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            # Groq models
            {
                "name": "llama2-70b",
                "provider": "groq",
                "model_id": "llama2-70b-4096",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            {
                "name": "mixtral-8x7b",
                "provider": "groq",
                "model_id": "mixtral-8x7b-32768",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            # DeepSeek models
            {
                "name": "deepseek-chat",
                "provider": "deepseek",
                "model_id": "deepseek-chat",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            {
                "name": "deepseek-coder",
                "provider": "deepseek",
                "model_id": "deepseek-coder",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            # SiliconFlow models
            {
                "name": "qwen2-72b",
                "provider": "siliconeflow",
                "model_id": "Qwen2-72B-Instruct",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
            # Moonshot models
            {
                "name": "moonshot-v1-8k",
                "provider": "moonshot",
                "model_id": "moonshot-v1-8k",
                "temperature": 0.7,
                "max_tokens": 8192,
                "enabled": True,
            },
            {
                "name": "moonshot-v1-32k",
                "provider": "moonshot",
                "model_id": "moonshot-v1-32k",
                "temperature": 0.7,
                "max_tokens": 32768,
                "enabled": True,
            },
            # Fireworks models
            {
                "name": "llama-v2-7b",
                "provider": "fireworks",
                "model_id": "accounts/fireworks/models/llama-v2-7b-chat",
                "temperature": 0.7,
                "max_tokens": 4096,
                "enabled": True,
            },
        ]

        for model_data in models_data:
            model = Model(
                name=model_data["name"],
                provider=model_data["provider"],
                model_id=model_data["model_id"],
                temperature=model_data["temperature"],
                max_tokens=model_data["max_tokens"],
                enabled=model_data["enabled"],
            )
            db.add(model)
        db.commit()
        logger.info("Models seeded successfully.")

    if db.query(SearchDocument).count() == 0:
        logger.info("Seeding initial search documents...")
        # Seed search documents
        search_docs = [
            {
                "id": "doc_1",
                "collection": "documents",
                "document": "This is a comprehensive guide to building modern web applications using React and TypeScript. It covers component architecture, state management, and best practices for scalable applications.",
                "metadata": {
                    "source": "docs",
                    "type": "guide",
                    "tags": ["react", "typescript", "web-development"],
                },
            },
            {
                "id": "doc_2",
                "collection": "documents",
                "document": "API design principles for microservices architecture. Learn about RESTful APIs, GraphQL, and how to design scalable backend services.",
                "metadata": {
                    "source": "docs",
                    "type": "tutorial",
                    "tags": ["api", "microservices", "backend"],
                },
            },
            {
                "id": "doc_3",
                "collection": "documents",
                "document": "Machine learning fundamentals including neural networks, deep learning, and practical applications in computer vision and natural language processing.",
                "metadata": {
                    "source": "docs",
                    "type": "reference",
                    "tags": ["ml", "ai", "neural-networks"],
                },
            },
            {
                "id": "code_1",
                "collection": "code",
                "document": "function calculateTotal(items) { return items.reduce((sum, item) => sum + item.price * item.quantity, 0); }",
                "metadata": {
                    "source": "code",
                    "language": "javascript",
                    "tags": ["javascript", "array-methods"],
                },
            },
            {
                "id": "code_2",
                "collection": "code",
                "document": "class UserService { constructor(db) { this.db = db; } async findById(id) { return this.db.users.find(u => u.id === id); } }",
                "metadata": {
                    "source": "code",
                    "language": "javascript",
                    "tags": ["javascript", "class", "service"],
                },
            },
            {
                "id": "kb_1",
                "collection": "knowledge",
                "document": "DevOps best practices include continuous integration, automated testing, infrastructure as code, and monitoring. These practices help teams deliver software faster and more reliably.",
                "metadata": {
                    "source": "knowledge",
                    "category": "devops",
                    "tags": ["devops", "ci-cd", "automation"],
                },
            },
            {
                "id": "kb_2",
                "collection": "knowledge",
                "document": "Security principles: defense in depth, least privilege, fail-safe defaults, and regular security audits are essential for protecting applications and data.",
                "metadata": {
                    "source": "knowledge",
                    "category": "security",
                    "tags": ["security", "best-practices", "auditing"],
                },
            },
        ]

        for doc_data in search_docs:
            collection_name = doc_data["collection"]
            collection_obj = db.query(SearchCollection).filter_by(name=collection_name).first()
            if not collection_obj:
                collection_obj = SearchCollection(name=collection_name)
                db.add(collection_obj)
                db.flush()  # Flush to get the ID for the new collection

            doc = SearchDocument(
                document_id=doc_data["id"],
                collection_id=collection_obj.id,
                document=doc_data["document"],
                metadata=doc_data["metadata"],
            )
            db.add(doc)
        db.commit()
        logger.info("Search documents seeded successfully.")

    if db.query(Task).count() == 0:
        logger.info("Seeding initial tasks...")
        # Seed some mock tasks
        mock_tasks = [
            {
                "id": "task_001",
                "goblin": "docs-writer",
                "task": "Write documentation for the new API endpoints",
                "status": "completed",
                "created_at": datetime.now(timezone.utc) - timedelta(hours=2),
                "updated_at": datetime.now(timezone.utc) - timedelta(hours=1, minutes=45),
                "result": "Successfully generated comprehensive API documentation with examples and usage patterns.",
            },
            {
                "id": "task_002",
                "goblin": "code-writer",
                "task": "Implement user authentication middleware",
                "status": "completed",
                "created_at": datetime.now(timezone.utc) - timedelta(hours=4),
                "updated_at": datetime.now(timezone.utc) - timedelta(hours=3, minutes=30),
                "result": "Created JWT-based authentication middleware with role-based access control and secure token handling.",
            },
            {
                "id": "task_003",
                "goblin": "docs-writer",
                "task": "Create deployment guide for the application",
                "status": "completed",
                "created_at": datetime.now(timezone.utc) - timedelta(hours=6),
                "updated_at": datetime.now(timezone.utc) - timedelta(hours=5, minutes=15),
                "result": "Compiled detailed deployment guide covering Docker, Kubernetes, and cloud platform configurations.",
            },
        ]

        for task_data in mock_tasks:
            task = Task(
                id=task_data["id"],
                user_id=None,
                goblin=task_data["goblin"],
                task=task_data["task"],
                status=task_data["status"],
                created_at=task_data["created_at"],
                updated_at=task_data["updated_at"],
                result=task_data["result"],
            )
            db.add(task)
        db.commit()
        logger.info("Tasks seeded successfully.")
