#!/usr/bin/env python3
"""
Cloud Provider Health Check Script

Tests connectivity and health for all cloud providers:
- GCP (Ollama/llama.cpp)
- RunPod (Serverless inference)
- Vast.ai (Spot instances)

Run with: python -m test_cloud_providers (from backend directory)
Or: PYTHONPATH=. python test_cloud_providers.py
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[2]

# Load environment from parent directory .env
env_path = BACKEND_DIR.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try local .env
    load_dotenv(BACKEND_DIR / ".env")

# Add backend to path for imports
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


async def test_gcp_provider():
    """Test GCP Ollama provider."""
    print("\n" + "=" * 60)
    print("🔍 Testing GCP Provider (Ollama)")
    print("=" * 60)

    try:
        from orchestrator.providers.gcp import GCPProvider

        provider = GCPProvider()
        print(f"   Ollama Host: {provider.ollama_host}")

        is_healthy = await provider.health_check()

        if is_healthy:
            print("   ✅ GCP Provider: HEALTHY")
            availability = await provider.get_availability()
            print(f"   📊 Availability: {availability}%")

            # Get cost estimate
            from orchestrator.providers.base import JobType

            cost = await provider.get_cost_estimate(JobType.INFERENCE)
            print(f"   💰 Cost estimate: ${cost}/hr")

            latency = await provider.get_latency_estimate(JobType.INFERENCE)
            print(f"   ⏱️  Latency estimate: {latency}ms")
        else:
            print("   ❌ GCP Provider: UNHEALTHY")
            print("      Ensure Ollama is running on the configured host")

        return is_healthy

    except Exception as e:
        print(f"   ❌ GCP Provider Error: {e}")
        return False


async def test_runpod_provider():
    """Test RunPod provider."""
    print("\n" + "=" * 60)
    print("🔍 Testing RunPod Provider")
    print("=" * 60)

    api_key = os.getenv("RUNPOD_API_KEY", "")
    if not api_key or api_key.startswith("rpa_") and len(api_key) < 20:
        print("   ⚠️  RUNPOD_API_KEY not set or invalid")
        print("      Get your key from: https://www.runpod.io/console/user/settings")
        return False

    print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")

    try:
        from orchestrator.providers.runpod import RunPodProvider

        provider = RunPodProvider()
        is_healthy = await provider.health_check()

        if is_healthy:
            print("   ✅ RunPod Provider: HEALTHY")
            availability = await provider.get_availability()
            print(f"   📊 Availability: {availability}%")

            from orchestrator.providers.base import JobType

            cost = await provider.get_cost_estimate(JobType.INFERENCE, "rtx_4090")
            print(f"   💰 RTX 4090 cost: ${cost}/hr")

            latency = await provider.get_latency_estimate(JobType.INFERENCE)
            print(f"   ⏱️  Latency estimate: {latency}ms")
        else:
            print("   ❌ RunPod Provider: UNHEALTHY")
            print("      Check API key and network connectivity")

        return is_healthy

    except Exception as e:
        print(f"   ❌ RunPod Provider Error: {e}")
        return False


async def test_vastai_provider():
    """Test Vast.ai provider."""
    print("\n" + "=" * 60)
    print("🔍 Testing Vast.ai Provider")
    print("=" * 60)

    api_key = os.getenv("VASTAI_API_KEY", "")
    if not api_key or len(api_key) < 20:
        print("   ⚠️  VASTAI_API_KEY not set or invalid")
        print("      Get your key from: https://vast.ai/console/account")
        return False

    print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")

    try:
        from orchestrator.providers.vastai import VastAIProvider

        provider = VastAIProvider()
        is_healthy = await provider.health_check()

        if is_healthy:
            print("   ✅ Vast.ai Provider: HEALTHY")
            availability = await provider.get_availability()
            print(f"   📊 Availability: {availability}%")

            from orchestrator.providers.base import JobType

            cost = await provider.get_cost_estimate(JobType.INFERENCE, "rtx_4090")
            print(f"   💰 RTX 4090 spot cost: ${cost}/hr")

            latency = await provider.get_latency_estimate(JobType.INFERENCE)
            print(f"   ⏱️  Latency estimate: {latency}ms")
        else:
            print("   ❌ Vast.ai Provider: UNHEALTHY")
            print("      Check API key and network connectivity")

        return is_healthy

    except Exception as e:
        print(f"   ❌ Vast.ai Provider Error: {e}")
        return False


async def test_model_storage():
    """Test GCS model storage connectivity."""
    print("\n" + "=" * 60)
    print("🔍 Testing Model Storage (GCS)")
    print("=" * 60)

    bucket = os.getenv("GCS_MODEL_BUCKET", "")
    project = os.getenv("GCS_PROJECT_ID", "")

    print(f"   Project: {project or 'NOT SET'}")
    print(f"   Bucket: {bucket or 'NOT SET'}")

    try:
        from services.model_storage import GCSModelStorage, GCS_AVAILABLE

        if not GCS_AVAILABLE:
            print("   ⚠️  google-cloud-storage not installed")
            print("      Run: pip install google-cloud-storage")
            return False

        storage = GCSModelStorage()

        # Check if we can list models
        models = await storage.list_models()
        print("   ✅ GCS Storage: CONNECTED")
        print(f"   📦 Models available: {len(models)}")

        for model in models[:5]:  # Show first 5
            print(f"      - {model.name} ({model.format.value})")

        return True

    except Exception as e:
        print(f"   ❌ GCS Storage Error: {e}")
        return False


async def test_orchestrator_router():
    """Test the orchestrator router."""
    print("\n" + "=" * 60)
    print("🔍 Testing Orchestrator Router")
    print("=" * 60)

    try:
        from orchestrator.router import ProviderRouter

        router = ProviderRouter()
        await router.initialize()

        print("   ✅ Router: INITIALIZED")

        # Check available providers
        print("   📡 Available Providers:")
        for provider_type, provider in router.providers.items():
            is_healthy = await provider.health_check()
            status = "✅ HEALTHY" if is_healthy else "❌ UNHEALTHY"
            print(f"      - {provider_type.value}: {status}")

        return True

    except Exception as e:
        print(f"   ❌ Router Error: {e}")
        return False


async def main():
    """Run all cloud provider tests."""
    print("\n" + "🚀 " * 20)
    print("GOBLIN ASSISTANT - CLOUD PROVIDER HEALTH CHECK")
    print("🚀 " * 20)

    results = {}

    # Test each provider
    results["GCP"] = await test_gcp_provider()
    results["RunPod"] = await test_runpod_provider()
    results["Vast.ai"] = await test_vastai_provider()
    results["GCS Storage"] = await test_model_storage()
    results["Router"] = await test_orchestrator_router()

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    healthy = sum(1 for v in results.values() if v)
    total = len(results)

    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {name}")

    print(f"\n   Total: {healthy}/{total} providers healthy")

    if healthy == total:
        print("\n   🎉 All providers are operational!")
    else:
        print("\n   ⚠️  Some providers need attention.")
        print("   Please check the errors above and fix configuration.")

    return healthy == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
