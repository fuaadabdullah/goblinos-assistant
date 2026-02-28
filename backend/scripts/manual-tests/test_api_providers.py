# pyright: reportMissingImports=false
#!/usr/bin/env python3
"""
GoblinOS Assistant - API Provider Test & Validation
Tests all configured LLM providers with correct API patterns.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

# Load environment variables
load_dotenv()

# Get API keys
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")


def test_openai():
    """Test OpenAI API (GPT-4, o3-mini, etc.)"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_KEY)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini for cost efficiency
            messages=[{"role": "user", "content": "Say 'OpenAI works!'"}],
            max_tokens=10,
        )

        result = resp.choices[0].message.content
        print(f"✅ OpenAI: {result}")
        return True
    except Exception as e:
        print(f"❌ OpenAI: {str(e)[:60]}")
        return False


def test_claude():
    """Test Anthropic Claude API"""
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=CLAUDE_KEY)

        resp = client.messages.create(
            model="claude-3-5-haiku-latest",  # Using haiku for cost efficiency
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'Claude works!'"}],
        )

        result = resp.content[0].text
        print(f"✅ Claude: {result}")
        return True
    except Exception as e:
        print(f"❌ Claude: {str(e)[:60]}")
        return False


def test_deepseek():
    """Test DeepSeek API (OpenAI-compatible with custom base URL)"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Say 'DeepSeek works!'"}],
            max_tokens=10,
        )

        result = resp.choices[0].message.content
        print(f"✅ DeepSeek: {result}")
        return True
    except Exception as e:
        print(f"❌ DeepSeek: {str(e)[:60]}")
        return False


def test_gemini():
    """Test Google Gemini API"""
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_KEY)

        model = genai.GenerativeModel("gemini-1.5-flash")

        resp = model.generate_content("Say 'Gemini works!'")
        result = resp.text
        print(f"✅ Gemini: {result}")
        return True
    except Exception as e:
        print(f"❌ Gemini: {str(e)[:60]}")
        return False


def ask(model_name: str, prompt: str) -> str:
    """
    Unified interface for all LLM providers.

    Args:
        model_name: "openai", "claude", "deepseek", or "gemini"
        prompt: The prompt to send to the model

    Returns:
        Model response as string
    """
    if model_name == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_KEY)
        return (
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            .choices[0]
            .message.content
        )

    if model_name == "claude":
        from anthropic import Anthropic

        client = Anthropic(api_key=CLAUDE_KEY)
        return (
            client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            .content[0]
            .text
        )

    if model_name == "deepseek":
        from openai import OpenAI

        client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")
        return (
            client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            .choices[0]
            .message.content
        )

    if model_name == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text

    raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    print("🧪 GoblinOS Assistant - API Provider Tests\n")
    print("=" * 60)

    results = {
        "OpenAI": test_openai(),
        "Claude": test_claude(),
        "DeepSeek": test_deepseek(),
        "Gemini": test_gemini(),
    }

    print("\n" + "=" * 60)
    print("📊 Results Summary:")
    print("=" * 60)

    working = sum(results.values())
    total = len(results)

    for provider, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"{emoji} {provider}")

    print(f"\n🎯 {working}/{total} providers working")

    if working == total:
        print("\n✅ All API providers are operational!")
    elif working > 0:
        print(f"\n⚠️  {total - working} provider(s) need valid API keys")
    else:
        print("\n❌ No providers are working - check your API keys!")
