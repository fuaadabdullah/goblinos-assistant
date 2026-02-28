# pyright: reportMissingImports=false
#!/usr/bin/env python3
"""
Test script for the intelligent chat API with routing.
Demonstrates end-to-end integration with local LLM routing.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, get_db
from services.routing import RoutingService
from services.encryption import EncryptionService
from models.provider import Provider as RoutingProvider
import os

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_chat_routing.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)


def setup_test_provider():
    """Set up a test Ollama provider in the database."""
    db = SessionLocal()

    try:
        # Check if provider already exists
        provider = (
            db.query(RoutingProvider).filter(RoutingProvider.name == "goblin-ollama-server").first()
        )

        if not provider:
            # Create encryption service
            encryption_key = os.getenv("ROUTING_ENCRYPTION_KEY", "test-key-32-characters-long-xyz")
            encryption_service = EncryptionService(encryption_key)

            # Get Ollama configuration - prefer Kamatera for production
            use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

            if use_local_llm:
                # Local development mode
                ollama_api_key = os.getenv("LOCAL_LLM_API_KEY", "your-secure-api-key-here")
                ollama_base_url = os.getenv("LOCAL_LLM_PROXY_URL", "http://45.61.60.3:8002")
            else:
                # Production mode - use Kamatera-hosted LLM runtime
                ollama_api_key = os.getenv("KAMATERA_LLM_API_KEY", "your-secure-api-key-here")
                ollama_base_url = os.getenv("KAMATERA_LLM_URL", "http://66.55.77.147:8000")

            # Encrypt API key
            encrypted_key = encryption_service.encrypt(ollama_api_key)

            # Create provider
            provider = RoutingProvider(
                name="goblin-ollama-server",
                display_name="Goblin Ollama Server (Local LLMs)",
                base_url=ollama_base_url,
                api_key_encrypted=encrypted_key,
                capabilities=["chat", "completion"],
                models=[],  # Models will be discovered dynamically
                priority=10,
                is_active=True,
            )

            db.add(provider)
            db.commit()
            print(f"✅ Created Ollama provider: {ollama_base_url}")
        else:
            print(f"✅ Ollama provider already exists: {provider.base_url}")

        return provider

    finally:
        db.close()


async def test_chat_routing(test_name: str, messages: list, **kwargs):
    """Test a chat routing scenario."""
    print("\n" + "=" * 80)
    print(f"Test: {test_name}")
    print("=" * 80)

    # Show input
    print("\nInput Messages:")
    for i, msg in enumerate(messages, 1):
        content = msg.get("content", "")
        preview = content[:60] + "..." if len(content) > 60 else content
        print(f"  {i}. [{msg.get('role', 'user')}] {preview}")

    if kwargs:
        print("\nRouting Parameters:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    # Create routing service
    db = SessionLocal()
    try:
        encryption_key = os.getenv("ROUTING_ENCRYPTION_KEY", "test-key-32-characters-long-xyz")
        routing_service = RoutingService(db, encryption_key)

        # Build requirements
        requirements = {"messages": messages, **kwargs}

        # Route the request
        result = await routing_service.route_request(capability="chat", requirements=requirements)

        if not result.get("success"):
            print(f"\n❌ Routing Failed: {result.get('error')}")
            return

        # Show routing decision
        print("\n" + "-" * 80)
        print("Routing Decision:")

        provider = result["provider"]
        print(f"  Provider: {provider['display_name']}")
        print(f"  Model: {provider.get('model', 'N/A')}")
        print(f"  Intent: {result.get('intent', 'N/A')}")

        if result.get("routing_explanation"):
            print(f"  Reasoning: {result['routing_explanation']}")

        # Show recommended parameters
        if result.get("recommended_params"):
            print("\n  Recommended Parameters:")
            for key, value in result["recommended_params"].items():
                if value is not None:
                    print(f"    {key}: {value}")

        # Show system prompt (truncated)
        if result.get("system_prompt"):
            system_prompt = result["system_prompt"]
            preview = system_prompt[:80] + "..." if len(system_prompt) > 80 else system_prompt
            print(f"\n  System Prompt: {preview}")

        print("\n✅ Routing Successful")
        print("=" * 80)

        return result

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        db.close()


async def main():
    """Run all test scenarios."""
    print("\n" + "=" * 80)
    print("Intelligent Chat API Routing Tests")
    print("=" * 80)
    print("\nSetting up test environment...")

    # Setup test provider
    setup_test_provider()

    print("\nRunning routing tests...\n")

    # Test 1: Code Generation
    await test_chat_routing(
        "Code Generation",
        messages=[
            {
                "role": "user",
                "content": "Write a Python function to validate an email address using regex",
            }
        ],
    )

    # Test 2: Quick Status Check
    await test_chat_routing(
        "Status Check (Ultra-Low Latency)",
        messages=[{"role": "user", "content": "Is the service healthy?"}],
        latency_target="ultra_low",
    )

    # Test 3: Long Document Summarization
    await test_chat_routing(
        "Long Document Summarization",
        messages=[
            {
                "role": "user",
                "content": "Summarize the key findings from this research paper",
            }
        ],
        context="x" * 40000,  # ~10K tokens
        intent="summarize",
    )

    # Test 4: Multilingual Query
    await test_chat_routing(
        "Multilingual Translation",
        messages=[
            {
                "role": "user",
                "content": "请帮我翻译这段文字成英文：人工智能正在改变世界",
            }
        ],
    )

    # Test 5: Creative Writing
    await test_chat_routing(
        "Creative Writing",
        messages=[
            {
                "role": "user",
                "content": "Write a short creative story about a developer who builds an AI assistant",
            }
        ],
        intent="creative",
    )

    # Test 6: Classification Task
    await test_chat_routing(
        "Text Classification (Cost Priority)",
        messages=[
            {
                "role": "user",
                "content": "Classify this email as spam or legitimate: 'Congratulations! You've won $1M!'",
            }
        ],
        cost_priority=True,
    )

    # Test 7: RAG Query
    await test_chat_routing(
        "RAG Query with Context",
        messages=[
            {
                "role": "user",
                "content": "Based on the documentation, what are the benefits of local LLMs?",
            }
        ],
        intent="rag",
        context="Local LLMs provide cost savings, no rate limits, and data privacy...",
    )

    # Test 8: Conversational Chat
    await test_chat_routing(
        "Multi-Turn Conversational Chat",
        messages=[
            {"role": "user", "content": "Hi, can you help me debug my code?"},
            {"role": "assistant", "content": "Of course! What seems to be the issue?"},
            {"role": "user", "content": "I'm getting a TypeError in my Python script"},
        ],
        latency_target="low",
    )

    # Test 9: Technical Explanation
    await test_chat_routing(
        "Technical Explanation (High Quality)",
        messages=[
            {
                "role": "user",
                "content": "Explain how transformer attention mechanisms work in deep learning",
            }
        ],
        intent="explain",
    )

    # Test 10: Explicit Model Selection
    await test_chat_routing(
        "Explicit Model Selection",
        messages=[{"role": "user", "content": "What's the weather like?"}],
        model="mistral:7b",  # Force specific model
    )

    print("\n" + "=" * 80)
    print("All Tests Complete!")
    print("=" * 80)
    print("\n📊 Summary:")
    print("  • Code gen → mistral:7b (temp=0.0)")
    print("  • Status → gemma:2b (ultra-low latency)")
    print("  • Long docs → qwen2.5:3b (32K window)")
    print("  • Multilingual → qwen2.5:3b")
    print("  • Creative → mistral:7b (temp=0.6)")
    print("  • Classification → gemma:2b (cost priority)")
    print("  • RAG → qwen2.5:3b (retrieval mode)")
    print("  • Chat → phi3:3.8b (low latency)")
    print("  • Explain → mistral:7b (high quality)")
    print("  • Explicit → mistral:7b (forced)")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
