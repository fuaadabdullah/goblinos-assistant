#!/usr/bin/env python3
"""Check what's actually loaded from .env file"""

import os
from dotenv import load_dotenv

load_dotenv()

keys = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
}

print("🔍 Environment Variable Check\n")
print("=" * 60)

for name, value in keys.items():
    if value:
        # Show first 15 and last 10 chars (more to see past potential quotes)
        masked = f"{value[:15]}...{value[-10:]}"
        length = len(value)
        has_newlines = "\n" in value
        has_spaces = " " in value
        starts_with = repr(value[:3])  # Show raw first 3 chars

        print(f"\n{name}:")
        print(f"  Length: {length} chars")
        print(f"  Preview: {masked}")
        print(f"  Starts with (raw): {starts_with}")
        print(f"  Has newlines: {has_newlines}")
        print(f"  Has spaces: {has_spaces}")

        # Check if it looks like a valid key
        if name == "OPENAI_API_KEY" and not value.startswith("sk-"):
            print(f"  ⚠️  WARNING: OpenAI keys should start with 'sk-'")
        if name == "ANTHROPIC_API_KEY" and not value.startswith("sk-"):
            print(f"  ⚠️  WARNING: Anthropic keys should start with 'sk-'")
        if name == "DEEPSEEK_API_KEY" and not value.startswith("sk-"):
            print(f"  ⚠️  WARNING: DeepSeek keys should start with 'sk-'")
        if name == "GEMINI_API_KEY" and not value.startswith("AIza"):
            print(f"  ⚠️  WARNING: Gemini keys should start with 'AIza'")
    else:
        print(f"\n{name}: ❌ NOT FOUND")

print("\n" + "=" * 60)
