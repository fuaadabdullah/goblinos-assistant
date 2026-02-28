# pyright: reportMissingImports=false
#!/usr/bin/env python3
"""
Test provider adapters directly to verify they work correctly.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

from providers.openai_adapter import OpenAIAdapter
from providers.anthropic_adapter import AnthropicAdapter
from providers.deepseek_adapter import DeepSeekAdapter
from providers.gemini_adapter import GeminiAdapter

# Load environment variables
load_dotenv()


async def test_openai():
    """Test OpenAI adapter."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"success": False, "error": "No API key found"}

    adapter = OpenAIAdapter(api_key=api_key, base_url=None)
    result = await adapter.test_completion(model="gpt-4o-mini", max_tokens=10)
    return result


async def test_anthropic():
    """Test Anthropic adapter."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"success": False, "error": "No API key found"}

    adapter = AnthropicAdapter(api_key=api_key, base_url=None)
    result = await adapter.test_completion(model="claude-3-5-haiku-20241022", max_tokens=10)
    return result


async def test_deepseek():
    """Test DeepSeek adapter."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return {"success": False, "error": "No API key found"}

    # Pass base_url=None to test the fix
    adapter = DeepSeekAdapter(api_key=api_key, base_url=None)
    result = await adapter.test_completion(model="deepseek-chat", max_tokens=10)
    return result


async def test_gemini():
    """Test Gemini adapter."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"success": False, "error": "No API key found"}

    adapter = GeminiAdapter(api_key=api_key, base_url=None)
    result = await adapter.test_completion(model="gemini-1.5-flash", max_tokens=10)
    return result


async def main():
    print("🧪 GoblinOS Assistant - Adapter Tests\n")
    print("=" * 60)

    # Test all providers
    tests = {
        "OpenAI": test_openai(),
        "Anthropic": test_anthropic(),
        "DeepSeek": test_deepseek(),
        "Gemini": test_gemini(),
    }

    results = {}
    for name, test_coro in tests.items():
        try:
            result = await test_coro
            results[name] = result

            if result.get("success"):
                print(f"✅ {name}: Success (response time: {result.get('response_time_ms')}ms)")
            else:
                error = result.get("error", "Unknown error")
                # Truncate long errors
                if len(error) > 60:
                    error = error[:60] + "..."
                print(f"❌ {name}: {error}")
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 60:
                error_msg = error_msg[:60] + "..."
            print(f"❌ {name}: {error_msg}")
            results[name] = {"success": False, "error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("📊 Results Summary:")
    print("=" * 60)

    successful = [name for name, result in results.items() if result.get("success")]
    failed = [name for name, result in results.items() if not result.get("success")]

    for name in successful:
        print(f"✅ {name}")
    for name in failed:
        print(f"❌ {name}")

    print(f"\n🎯 {len(successful)}/{len(results)} adapters working")

    if len(successful) == 0:
        print("\n❌ No adapters are working - check your API keys!")
    elif len(successful) < len(results):
        print(f"\n⚠️  {len(failed)} adapter(s) failing - check those API keys")
    else:
        print("\n✅ All adapters working correctly!")


if __name__ == "__main__":
    asyncio.run(main())
