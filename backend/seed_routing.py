#!/usr/bin/env python3
"""
Seed script to populate default routing providers in the database.
"""

import os
import sys
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from database import SessionLocal, create_tables
from models.provider import Provider as RoutingProvider
from services.encryption import EncryptionService

# Default provider configurations
DEFAULT_PROVIDERS = [
    {
        "name": "openai",
        "display_name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "capabilities": ["chat", "embeddings", "vision"],
        "models": [
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "capabilities": ["chat"],
                "context_window": 8192,
                "pricing": {"input": 0.03, "output": 0.06},
            },
            {
                "id": "gpt-4-turbo",
                "name": "GPT-4 Turbo",
                "capabilities": ["chat", "vision"],
                "context_window": 128000,
                "pricing": {"input": 0.01, "output": 0.03},
            },
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "capabilities": ["chat"],
                "context_window": 4096,
                "pricing": {"input": 0.0015, "output": 0.002},
            },
            {
                "id": "text-embedding-ada-002",
                "name": "Ada Embedding",
                "capabilities": ["embeddings"],
                "context_window": 8191,
                "pricing": {"input": 0.0001, "output": 0.0001},
            },
        ],
        "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 40000},
        "cost_per_token": 0.00002,  # Average cost per token
        "priority": 10,
        "is_active": True,
    },
    {
        "name": "anthropic",
        "display_name": "Anthropic",
        "base_url": "https://api.anthropic.com",
        "capabilities": ["chat", "vision"],
        "models": [
            {
                "id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "capabilities": ["chat", "vision"],
                "context_window": 200000,
                "pricing": {"input": 0.015, "output": 0.075},
            },
            {
                "id": "claude-3-sonnet-20240229",
                "name": "Claude 3 Sonnet",
                "capabilities": ["chat", "vision"],
                "context_window": 200000,
                "pricing": {"input": 0.003, "output": 0.015},
            },
            {
                "id": "claude-3-haiku-20240307",
                "name": "Claude 3 Haiku",
                "capabilities": ["chat", "vision"],
                "context_window": 200000,
                "pricing": {"input": 0.00025, "output": 0.00125},
            },
            {
                "id": "claude-2.1",
                "name": "Claude 2.1",
                "capabilities": ["chat"],
                "context_window": 200000,
                "pricing": {"input": 0.008, "output": 0.024},
            },
        ],
        "rate_limits": {"requests_per_minute": 50, "tokens_per_minute": 25000},
        "cost_per_token": 0.000015,  # Average cost per token
        "priority": 9,
        "is_active": True,
    },
    {
        "name": "gemini",
        "display_name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com",
        "capabilities": ["chat", "vision"],
        "models": [
            {
                "id": "gemini-pro",
                "name": "Gemini Pro",
                "capabilities": ["chat"],
                "context_window": 32768,
                "pricing": {"input": 0.00025, "output": 0.0005},
            },
            {
                "id": "gemini-pro-vision",
                "name": "Gemini Pro Vision",
                "capabilities": ["chat", "vision"],
                "context_window": 16384,
                "pricing": {"input": 0.00025, "output": 0.0005},
            },
            {
                "id": "gemini-1.5-pro",
                "name": "Gemini 1.5 Pro",
                "capabilities": ["chat", "vision"],
                "context_window": 2097152,
                "pricing": {"input": 0.00125, "output": 0.005},
            },
            {
                "id": "gemini-1.5-flash",
                "name": "Gemini 1.5 Flash",
                "capabilities": ["chat", "vision"],
                "context_window": 1048576,
                "pricing": {"input": 0.000075, "output": 0.0003},
            },
        ],
        "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 32000},
        "cost_per_token": 0.000001,  # Average cost per token
        "priority": 8,
        "is_active": True,
    },
    {
        "name": "grok",
        "display_name": "Grok (xAI)",
        "base_url": "https://api.x.ai/v1",
        "capabilities": ["chat", "vision"],
        "models": [
            {
                "id": "grok-beta",
                "name": "Grok Beta",
                "capabilities": ["chat"],
                "context_window": 128000,
                "pricing": {"input": 0.005, "output": 0.015},
            },
            {
                "id": "grok-vision-beta",
                "name": "Grok Vision Beta",
                "capabilities": ["chat", "vision"],
                "context_window": 128000,
                "pricing": {"input": 0.005, "output": 0.015},
            },
        ],
        "rate_limits": {"requests_per_minute": 30, "tokens_per_minute": 15000},
        "cost_per_token": 0.00001,  # Average cost per token
        "priority": 7,
        "is_active": True,
    },
    {
        "name": "deepseek",
        "display_name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "capabilities": ["chat", "code"],
        "models": [
            {
                "id": "deepseek-chat",
                "name": "DeepSeek Chat",
                "capabilities": ["chat"],
                "context_window": 32768,
                "pricing": {"input": 0.00014, "output": 0.00028},
            },
            {
                "id": "deepseek-coder",
                "name": "DeepSeek Coder",
                "capabilities": ["chat", "code"],
                "context_window": 32768,
                "pricing": {"input": 0.00014, "output": 0.00028},
            },
            {
                "id": "deepseek-chat-67b",
                "name": "DeepSeek Chat 67B",
                "capabilities": ["chat"],
                "context_window": 4096,
                "pricing": {"input": 0.00014, "output": 0.00028},
            },
            {
                "id": "deepseek-coder-33b",
                "name": "DeepSeek Coder 33B",
                "capabilities": ["chat", "code"],
                "context_window": 16384,
                "pricing": {"input": 0.00014, "output": 0.00028},
            },
        ],
        "rate_limits": {"requests_per_minute": 100, "tokens_per_minute": 60000},
        "cost_per_token": 0.00000021,  # Average cost per token
        "priority": 6,
        "is_active": True,
    },
    {
        "name": "azure_openai",
        "display_name": "Azure OpenAI",
        "base_url": os.getenv(
            "AZURE_OPENAI_ENDPOINT", "https://{resource}.openai.azure.com"
        ),
        "capabilities": ["chat", "code", "embeddings"],
        "models": [
            {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "capabilities": ["chat"],
                "context_window": 128000,
                "pricing": {"input": 0.005, "output": 0.015},
            },
            {
                "id": "gpt-4.1",
                "name": "GPT-4.1",
                "capabilities": ["chat", "code"],
                "context_window": 128000,
                "pricing": {"input": 0.002, "output": 0.008},
            },
            {
                "id": "gpt-4-turbo",
                "name": "GPT-4 Turbo",
                "capabilities": ["chat", "code"],
                "context_window": 128000,
                "pricing": {"input": 0.01, "output": 0.03},
            },
        ],
        "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 60000},
        "cost_per_token": 0.000006,  # Average cost per token
        "priority": 16,
        "is_active": True,
    },
    {
        "name": "aliyun",
        "display_name": "Aliyun",
        "base_url": os.getenv(
            "ALIYUN_MODEL_SERVER_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "capabilities": ["chat", "code"],
        "models": [
            {
                "id": "qwen-plus",
                "name": "Qwen Plus",
                "capabilities": ["chat"],
                "context_window": 131072,
                "pricing": {"input": 0.0008, "output": 0.002},
            },
            {
                "id": "qwen-max",
                "name": "Qwen Max",
                "capabilities": ["chat"],
                "context_window": 32768,
                "pricing": {"input": 0.002, "output": 0.006},
            },
            {
                "id": "qwen2.5-coder-32b-instruct",
                "name": "Qwen 2.5 Coder 32B Instruct",
                "capabilities": ["chat", "code"],
                "context_window": 32768,
                "pricing": {"input": 0.0008, "output": 0.002},
            },
        ],
        "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 50000},
        "cost_per_token": 0.0000014,  # Average cost per token
        "priority": 16,
        "is_active": True,
    },
    {
        "name": "ollama",
        "display_name": "Ollama (Local)",
        "base_url": "http://localhost:8002",  # Points to local proxy
        "capabilities": ["chat"],
        "models": [
            {
                "id": "phi3:3.8b",
                "name": "Phi-3 3.8B",
                "capabilities": ["chat"],
                "context_window": 4096,
                "pricing": {"input": 0.0, "output": 0.0},  # Free local model
            },
            {
                "id": "gemma:2b",
                "name": "Gemma 2B",
                "capabilities": ["chat"],
                "context_window": 8192,
                "pricing": {"input": 0.0, "output": 0.0},
            },
            {
                "id": "qwen2.5:3b",
                "name": "Qwen 2.5 3B",
                "capabilities": ["chat"],
                "context_window": 32768,
                "pricing": {"input": 0.0, "output": 0.0},
            },
            {
                "id": "deepseek-coder:1.3b",
                "name": "DeepSeek Coder 1.3B",
                "capabilities": ["chat", "code"],
                "context_window": 16384,
                "pricing": {"input": 0.0, "output": 0.0},
            },
        ],
        "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 30000},
        "cost_per_token": 0.0,  # Free local inference
        "priority": 15,  # High priority for local models
        "is_active": True,
    },
    {
        "name": "llamacpp",
        "display_name": "Llama.cpp (Local)",
        "base_url": "http://localhost:8002",  # Points to local proxy
        "capabilities": ["chat"],
        "models": [
            {
                "id": "active-model",
                "name": "Active Model",
                "capabilities": ["chat"],
                "context_window": 4096,
                "pricing": {"input": 0.0, "output": 0.0},  # Free local model
            },
        ],
        "rate_limits": {"requests_per_minute": 30, "tokens_per_minute": 15000},
        "cost_per_token": 0.0,  # Free local inference
        "priority": 14,  # High priority for local models
        "is_active": True,
    },
    {
        "name": "ollama_gcp",
        "display_name": "Ollama (GCP)",
        "base_url": "http://YOUR_GCP_OLLAMA_IP:8002",  # Update with actual GCP endpoint
        "capabilities": ["chat", "reasoning", "code", "embedding"],
        "models": [
            {
                "id": "phi3:3.8b",
                "name": "Phi-3 3.8B",
                "capabilities": ["chat"],
                "context_window": 4096,
                "pricing": {"input": 0.0, "output": 0.0},  # Free self-hosted model
            },
            {
                "id": "gemma:2b",
                "name": "Gemma 2B",
                "capabilities": ["chat"],
                "context_window": 8192,
                "pricing": {"input": 0.0, "output": 0.0},
            },
            {
                "id": "qwen2.5:3b",
                "name": "Qwen 2.5 3B",
                "capabilities": ["chat"],
                "context_window": 32768,
                "pricing": {"input": 0.0, "output": 0.0},
            },
            {
                "id": "deepseek-coder:1.3b",
                "name": "DeepSeek Coder 1.3B",
                "capabilities": ["chat", "code"],
                "context_window": 16384,
                "pricing": {"input": 0.0, "output": 0.0},
            },
            {
                "id": "mistral:7b",
                "name": "Mistral 7B",
                "capabilities": ["chat"],
                "context_window": 4096,
                "pricing": {"input": 0.0, "output": 0.0},
            },
        ],
        "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 30000},
        "cost_per_token": 0.0,  # Free self-hosted inference
        "priority": 16,  # High priority for self-hosted models
        "is_active": True,
    },
    {
        "name": "llamacpp_gcp",
        "display_name": "Llama.cpp (GCP)",
        "base_url": "http://YOUR_GCP_LLAMACPP_IP:8002",  # Update with actual GCP endpoint
        "capabilities": ["chat", "reasoning", "code"],
        "models": [
            {
                "id": "phi-3-mini-4k-instruct-q4",
                "name": "Phi-3 Mini 4K Instruct Q4",
                "capabilities": ["chat"],
                "context_window": 4096,
                "pricing": {"input": 0.0, "output": 0.0},  # Free self-hosted model
            },
            {
                "id": "llama-2-7b-chat-q4_k_m",
                "name": "Llama 2 7B Chat Q4_K_M",
                "capabilities": ["chat"],
                "context_window": 4096,
                "pricing": {"input": 0.0, "output": 0.0},
            },
            {
                "id": "mistral-7b-instruct-v0.2-q4_k_m",
                "name": "Mistral 7B Instruct v0.2 Q4_K_M",
                "capabilities": ["chat"],
                "context_window": 4096,
                "pricing": {"input": 0.0, "output": 0.0},
            },
        ],
        "rate_limits": {"requests_per_minute": 30, "tokens_per_minute": 15000},
        "cost_per_token": 0.0,  # Free self-hosted inference
        "priority": 15,  # High priority for self-hosted models
        "is_active": True,
    },
    {
        "name": "groq",
        "display_name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "capabilities": ["chat"],
        "models": [
            {
                "id": "llama-3.3-70b-versatile",
                "name": "Llama 3.3 70B",
                "capabilities": ["chat"],
                "context_window": 32768,
                "pricing": {"input": 0.00059, "output": 0.00079},
            },
            {
                "id": "llama-3.1-70b-versatile",
                "name": "Llama 3.1 70B",
                "capabilities": ["chat"],
                "context_window": 131072,
                "pricing": {"input": 0.00059, "output": 0.00079},
            },
            {
                "id": "llama-3.1-8b-instant",
                "name": "Llama 3.1 8B Instant",
                "capabilities": ["chat"],
                "context_window": 131072,
                "pricing": {"input": 0.00005, "output": 0.00008},
            },
            {
                "id": "mixtral-8x7b-32768",
                "name": "Mixtral 8x7B",
                "capabilities": ["chat"],
                "context_window": 32768,
                "pricing": {"input": 0.00027, "output": 0.00027},
            },
            {
                "id": "gemma2-9b-it",
                "name": "Gemma 2 9B",
                "capabilities": ["chat"],
                "context_window": 8192,
                "pricing": {"input": 0.0002, "output": 0.0002},
            },
        ],
        "rate_limits": {"requests_per_minute": 30, "tokens_per_minute": 20000},
        "cost_per_token": 0.0005,  # Average cost per token
        "priority": 13,
        "is_active": True,
    },
    {
        "name": "mistral",
        "display_name": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "capabilities": ["chat", "embeddings"],
        "models": [
            {
                "id": "mistral-large-latest",
                "name": "Mistral Large",
                "capabilities": ["chat"],
                "context_window": 128000,
                "pricing": {"input": 0.002, "output": 0.006},
            },
            {
                "id": "mistral-medium-latest",
                "name": "Mistral Medium",
                "capabilities": ["chat"],
                "context_window": 32000,
                "pricing": {"input": 0.0027, "output": 0.0081},
            },
            {
                "id": "mistral-small-latest",
                "name": "Mistral Small",
                "capabilities": ["chat"],
                "context_window": 32000,
                "pricing": {"input": 0.0002, "output": 0.0006},
            },
            {
                "id": "mistral-embed",
                "name": "Mistral Embed",
                "capabilities": ["embeddings"],
                "context_window": 8192,
                "pricing": {"input": 0.0001, "output": 0.0001},
            },
        ],
        "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 50000},
        "cost_per_token": 0.000002,  # Average cost per token
        "priority": 12,
        "is_active": True,
    },
    {
        "name": "cohere",
        "display_name": "Cohere",
        "base_url": "https://api.cohere.ai/v1",
        "capabilities": ["chat", "embeddings", "rerank"],
        "models": [
            {
                "id": "command-r-plus",
                "name": "Command R+",
                "capabilities": ["chat"],
                "context_window": 128000,
                "pricing": {"input": 0.003, "output": 0.015},
            },
            {
                "id": "command-r",
                "name": "Command R",
                "capabilities": ["chat"],
                "context_window": 128000,
                "pricing": {"input": 0.0005, "output": 0.0015},
            },
            {
                "id": "command",
                "name": "Command",
                "capabilities": ["chat"],
                "context_window": 4096,
                "pricing": {"input": 0.001, "output": 0.002},
            },
            {
                "id": "embed-english-v3.0",
                "name": "Embed English v3",
                "capabilities": ["embeddings"],
                "context_window": 512,
                "pricing": {"input": 0.0001, "output": 0.0001},
            },
        ],
        "rate_limits": {"requests_per_minute": 100, "tokens_per_minute": 60000},
        "cost_per_token": 0.000003,  # Average cost per token
        "priority": 11,
        "is_active": True,
    },
    {
        "name": "perplexity",
        "display_name": "Perplexity AI",
        "base_url": "https://api.perplexity.ai",
        "capabilities": ["chat", "search"],
        "models": [
            {
                "id": "llama-3.1-sonar-large-128k-online",
                "name": "Sonar Large Online",
                "capabilities": ["chat", "search"],
                "context_window": 127072,
                "pricing": {"input": 0.001, "output": 0.001},
            },
            {
                "id": "llama-3.1-sonar-small-128k-online",
                "name": "Sonar Small Online",
                "capabilities": ["chat", "search"],
                "context_window": 127072,
                "pricing": {"input": 0.0002, "output": 0.0002},
            },
            {
                "id": "llama-3.1-70b-instruct",
                "name": "Llama 3.1 70B",
                "capabilities": ["chat"],
                "context_window": 131072,
                "pricing": {"input": 0.001, "output": 0.001},
            },
        ],
        "rate_limits": {"requests_per_minute": 50, "tokens_per_minute": 40000},
        "cost_per_token": 0.000001,  # Average cost per token
        "priority": 10,
        "is_active": True,
    },
]


def seed_routing_providers():
    """Seed the database with default routing providers."""
    # Get encryption key
    encryption_key = os.getenv("ROUTING_ENCRYPTION_KEY")
    if not encryption_key:
        print("Error: ROUTING_ENCRYPTION_KEY environment variable not set")
        sys.exit(1)

    encryption_service = EncryptionService(encryption_key)

    db = SessionLocal()
    try:
        # Create tables if they don't exist
        create_tables()

        for provider_config in DEFAULT_PROVIDERS:
            # Check if provider already exists
            existing = (
                db.query(RoutingProvider)
                .filter(RoutingProvider.name == provider_config["name"])
                .first()
            )

            if existing:
                print(f"Provider {provider_config['name']} already exists, skipping...")
                continue

            # Generate a placeholder encrypted API key (will need to be updated with real keys)
            placeholder_key = f"placeholder_{provider_config['name']}_key"
            encrypted_key = encryption_service.encrypt(placeholder_key)

            # Create provider
            provider = RoutingProvider(
                name=provider_config["name"],
                display_name=provider_config["display_name"],
                base_url=provider_config["base_url"],
                api_key_encrypted=encrypted_key,
                is_active=provider_config["is_active"],
                capabilities=provider_config["capabilities"],
                models=provider_config["models"],
                rate_limits=provider_config["rate_limits"],
                cost_per_token=provider_config["cost_per_token"],
                priority=provider_config["priority"],
            )

            db.add(provider)
            print(f"Added provider: {provider_config['name']}")

        db.commit()
        print("Successfully seeded routing providers!")

        # Print instructions for updating API keys
        print("\n" + "=" * 50)
        print("IMPORTANT: Update API keys for the seeded providers!")
        print("=" * 50)
        print("The providers have been created with placeholder API keys.")
        print("You need to update them with real API keys using one of these methods:")
        print()
        print("1. Using the settings API:")
        print("   PUT /settings/providers/{provider_name}")
        print('   Body: {"api_key": "your-real-api-key"}')
        print()
        print("2. Directly in the database:")
        print("   Update the api_key_encrypted field in routing_providers table")
        print()
        print("3. Using a custom script to update the keys")
        print("=" * 50)

    except Exception as e:
        print(f"Error seeding providers: {e}")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    print("Seeding routing providers...")
    seed_routing_providers()
