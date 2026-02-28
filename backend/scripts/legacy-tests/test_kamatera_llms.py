"""
Test script for Ollama and Llama.cpp adapters connecting to Kamatera VPS.
Tests the deployed local LLM infrastructure.
"""

import asyncio
import os
import sys
import time
import tomllib
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]

# Add backend to path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# Load environment variables
def load_env_direct():
    """Load .env file directly to bypass VS Code redaction."""
    env_path = BACKEND_DIR / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


def load_provider_config():
    """Load provider configuration from providers.toml."""
    config_path = BACKEND_DIR.parent / "config" / "providers.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        return config.get("providers", {})
    return {}


load_env_direct()

from providers.ollama_adapter import OllamaAdapter
from providers.llamacpp_adapter import LlamaCppAdapter


async def test_ollama_health():
    """Test Ollama health check via Kamatera proxy."""
    print("\n" + "=" * 60)
    print("Testing Ollama Health Check (via Kamatera)")
    print("=" * 60)

    try:
        api_key = os.getenv("LOCAL_LLM_API_KEY", "your-secure-api-key-here")
        base_url = os.getenv("LOCAL_LLM_PROXY_URL", "http://45.61.60.3:8002")

        print(f"API Key: {api_key[:20]}... ({len(api_key)} chars)")
        print(f"Base URL: {base_url}")

        adapter = OllamaAdapter(api_key=api_key, base_url=base_url)

        start = time.time()
        health = await adapter.health_check()
        elapsed = (time.time() - start) * 1000

        print(f"\n✅ Health Check Results:")
        print(f"  Healthy: {health.get('healthy')}")
        print(f"  Response Time: {elapsed:.2f}ms")
        print(f"  Available Models: {health.get('available_models', 0)}")
        print(f"  Error Rate: {health.get('error_rate', 0)}")

        return health.get("healthy", False)
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_ollama_list_models():
    """Test listing Ollama models."""
    print("\n" + "=" * 60)
    print("Testing Ollama Model Listing")
    print("=" * 60)

    try:
        api_key = os.getenv("LOCAL_LLM_API_KEY", "your-secure-api-key-here")
        base_url = os.getenv("LOCAL_LLM_PROXY_URL", "http://45.61.60.3:8002")

        adapter = OllamaAdapter(api_key=api_key, base_url=base_url)
        models = await adapter.list_models()

        print(f"\n✅ Found {len(models)} Ollama models:")
        for model in models:
            print(f"\n  Model: {model['name']}")
            print(f"    ID: {model['id']}")
            print(f"    Context Window: {model['context_window']:,} tokens")
            print(f"    Capabilities: {', '.join(model['capabilities'])}")
            print(
                f"    Pricing: ${model['pricing']['input']}/token input, ${model['pricing']['output']}/token output"
            )
            print(f"    Local: {model['local']}")

        return len(models) > 0
    except Exception as e:
        print(f"❌ Model Listing Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_ollama_completion():
    """Test Ollama completion."""
    print("\n" + "=" * 60)
    print("Testing Ollama Completion (phi3:3.8b)")
    print("=" * 60)

    try:
        api_key = os.getenv("LOCAL_LLM_API_KEY", "your-secure-api-key-here")
        base_url = os.getenv("LOCAL_LLM_PROXY_URL", "http://45.61.60.3:8002")

        # Load invoke_path from providers.toml
        provider_config = load_provider_config()
        invoke_path = provider_config.get("ollama_kamatera", {}).get("invoke_path", "/api/chat")

        adapter = OllamaAdapter(api_key=api_key, base_url=base_url, invoke_path=invoke_path)

        print("\nSending test prompt...")
        result = await adapter.test_completion(model="phi3:3.8b", max_tokens=50)

        print("\n✅ Completion Test Results:")
        print(f"  Success: {result.get('success')}")
        print(f"  Response Time: {result.get('response_time_ms'):.2f}ms")
        print(f"  Model: {result.get('model')}")
        print(f"  Local: {result.get('local')}")

        if not result.get("success"):
            print(f"  Error: {result.get('error')}")

        return result.get("success", False)
    except Exception as e:
        print(f"❌ Completion Test Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_llamacpp_health():
    """Test Llama.cpp health check via Kamatera proxy."""
    print("\n" + "=" * 60)
    print("Testing Llama.cpp Health Check (via Kamatera)")
    print("=" * 60)

    try:
        api_key = os.getenv("LOCAL_LLM_API_KEY", "your-secure-api-key-here")
        base_url = os.getenv("LOCAL_LLM_PROXY_URL", "http://45.61.60.3:8002")

        adapter = LlamaCppAdapter(api_key=api_key, base_url=base_url)

        start = time.time()
        health = await adapter.health_check()
        elapsed = (time.time() - start) * 1000

        print(f"\n✅ Health Check Results:")
        print(f"  Healthy: {health.get('healthy')}")
        print(f"  Response Time: {elapsed:.2f}ms")
        print(f"  Available Models: {health.get('available_models', 0)}")
        print(f"  Error Rate: {health.get('error_rate', 0)}")

        return health.get("healthy", False)
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_llamacpp_list_models():
    """Test listing Llama.cpp models."""
    print("\n" + "=" * 60)
    print("Testing Llama.cpp Model Listing")
    print("=" * 60)

    try:
        api_key = os.getenv("LOCAL_LLM_API_KEY", "your-secure-api-key-here")
        base_url = os.getenv("LOCAL_LLM_PROXY_URL", "http://45.61.60.3:8002")

        adapter = LlamaCppAdapter(api_key=api_key, base_url=base_url)
        models = await adapter.list_models()

        print(f"\n✅ Found {len(models)} Llama.cpp models:")
        for model in models:
            print(f"\n  Model: {model['name']}")
            print(f"    ID: {model['id']}")
            print(f"    Context Window: {model['context_window']:,} tokens")
            print(f"    Capabilities: {', '.join(model['capabilities'])}")
            print(f"    Local: {model['local']}")

        return len(models) > 0
    except Exception as e:
        print(f"❌ Model Listing Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_full_conversation():
    """Test a full conversation with Ollama."""
    print("\n" + "=" * 60)
    print("Testing Full Conversation (qwen2.5:3b)")
    print("=" * 60)

    try:
        api_key = os.getenv("LOCAL_LLM_API_KEY", "your-secure-api-key-here")
        base_url = os.getenv("LOCAL_LLM_PROXY_URL", "http://45.61.60.3:8002")

        # Load invoke_path from providers.toml
        provider_config = load_provider_config()
        invoke_path = provider_config.get("ollama_kamatera", {}).get("invoke_path", "/api/chat")

        adapter = OllamaAdapter(api_key=api_key, base_url=base_url, invoke_path=invoke_path)

        print("\nPrompt: What are the key features of FastAPI?")
        result = await adapter.test_completion(model="qwen2.5:3b", max_tokens=100)

        print("\n✅ Conversation Test:")
        print(f"  Success: {result.get('success')}")
        print(f"  Response Time: {result.get('response_time_ms'):.2f}ms")
        print(f"  Model: {result.get('model')}")

        return result.get("success", False)
    except Exception as e:
        print(f"❌ Conversation Test Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("KAMATERA LOCAL LLM DEPLOYMENT TEST SUITE")
    print("=" * 60)

    # Check environment
    api_key = os.getenv("LOCAL_LLM_API_KEY")
    base_url = os.getenv("LOCAL_LLM_PROXY_URL")

    if not api_key or not base_url:
        print("❌ ERROR: LOCAL_LLM_API_KEY or LOCAL_LLM_PROXY_URL not found in .env")
        print(f"API Key: {api_key}")
        print(f"Base URL: {base_url}")
        return

    print(f"\nConfiguration:")
    print(f"  Proxy URL: {base_url}")
    print(f"  API Key: {api_key[:20]}... ({len(api_key)} chars)")

    # Run tests
    tests = [
        ("Ollama Health Check", test_ollama_health),
        ("Ollama Model Listing", test_ollama_list_models),
        ("Ollama Completion Test", test_ollama_completion),
        ("Llama.cpp Health Check", test_llamacpp_health),
        ("Llama.cpp Model Listing", test_llamacpp_list_models),
        ("Full Conversation Test", test_full_conversation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Kamatera LLM deployment is working correctly.")
        print("\nDeployed Models:")
        print("  • phi3:3.8b - Fast general-purpose model (2.2GB)")
        print("  • qwen2.5:3b - Excellent multilingual model (1.9GB)")
        print("\nServer Info:")
        print("  • Location: Kamatera VPS (Atlanta, US)")
        print("  • IP: 45.61.60.3")
        print("  • RAM: 10GB")
        print("  • Disk: 20GB")
        print("  • Services: Ollama + Local LLM Proxy")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("  1. Verify server is running: ssh -i ~/kamatera_key root@45.61.60.3")
        print("  2. Check services: systemctl status ollama local-llm-proxy")
        print("  3. Test directly: curl http://45.61.60.3:8002/health")
        print("  4. Check firewall: ufw status")


if __name__ == "__main__":
    asyncio.run(main())
