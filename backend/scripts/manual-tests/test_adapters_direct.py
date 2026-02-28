# pyright: reportMissingImports=false
#!/usr/bin/env python3
"""
Test provider adapters with direct .env file reading (bypassing os.getenv redaction)
"""

import asyncio
import re
import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

from providers.openai_adapter import OpenAIAdapter
from providers.anthropic_adapter import AnthropicAdapter
from providers.deepseek_adapter import DeepSeekAdapter
from providers.gemini_adapter import GeminiAdapter


def load_env_direct():
    """Load .env file directly without using os.getenv()"""
    env_path = BACKEND_DIR / ".env"
    keys = {}
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    match = re.match(r"([A-Z_]+)=(.*)", line)
                    if match:
                        key, value = match.groups()
                        keys[key] = value
    return keys


async def test_openai(api_key):
    """Test OpenAI adapter."""
    if not api_key or len(api_key) < 20:
        return {"success": False, "error": "Invalid API key"}

    adapter = OpenAIAdapter(api_key=api_key, base_url=None)
    result = await adapter.test_completion(model="gpt-4o-mini", max_tokens=10)
    return result


async def test_anthropic(api_key):
    """Test Anthropic adapter."""
    if not api_key or len(api_key) < 20:
        return {"success": False, "error": "Invalid API key"}

    adapter = AnthropicAdapter(api_key=api_key, base_url=None)
    result = await adapter.test_completion(model="claude-3-5-haiku-20241022", max_tokens=10)
    return result


async def test_deepseek(api_key):
    """Test DeepSeek adapter."""
    if not api_key or len(api_key) < 20:
        return {"success": False, "error": "Invalid API key"}

    adapter = DeepSeekAdapter(api_key=api_key, base_url=None)
    result = await adapter.test_completion(model="deepseek-chat", max_tokens=10)
    return result


async def test_gemini(api_key):
    """Test Gemini adapter."""
    if not api_key or len(api_key) < 20:
        return {"success": False, "error": "Invalid API key"}

    adapter = GeminiAdapter(api_key=api_key, base_url=None)
    result = await adapter.test_completion(model="gemini-2.5-flash", max_tokens=50)
    return result


async def main():
    print("🧪 GoblinOS Assistant - Direct Adapter Tests\n")
    print("=" * 60)
    print("📝 Loading API keys directly from .env file...\n")

    # Load keys directly
    env_keys = load_env_direct()

    # Verify keys loaded
    key_status = {}
    for key_name in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
    ]:
        value = env_keys.get(key_name, "")
        key_status[key_name] = len(value) > 20
        status = "✅" if key_status[key_name] else "❌"
        print(f"{status} {key_name}: {len(value)} chars")

    print("\n" + "=" * 60)
    print("🧪 Running adapter tests...\n")

    # Test all providers
    tests = {
        "OpenAI": test_openai(env_keys.get("OPENAI_API_KEY", "")),
        "Anthropic": test_anthropic(env_keys.get("ANTHROPIC_API_KEY", "")),
        "DeepSeek": test_deepseek(env_keys.get("DEEPSEEK_API_KEY", "")),
        "Gemini": test_gemini(env_keys.get("GEMINI_API_KEY", "")),
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
