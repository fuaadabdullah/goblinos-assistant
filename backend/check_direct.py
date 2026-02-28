#!/usr/bin/env python3
"""Test reading .env file directly without dotenv library"""

import re

# Read .env file directly
with open(".env", "r") as f:
    content = f.read()

# Parse keys manually
keys = {}
for line in content.split("\n"):
    if line.strip() and not line.strip().startswith("#"):
        match = re.match(r"([A-Z_]+)=(.*)", line)
        if match:
            key, value = match.groups()
            if "API_KEY" in key:
                keys[key] = value

print("🔍 Direct File Read Check\n")
print("=" * 60)

for name, value in keys.items():
    if value:
        masked = f"{value[:15]}...{value[-10:]}"
        length = len(value)
        print(f"\n{name}:")
        print(f"  Length: {length} chars")
        print(f"  Preview: {masked}")
        print(f"  Valid format: {'✅' if length > 20 else '❌'}")

print("\n" + "=" * 60)
