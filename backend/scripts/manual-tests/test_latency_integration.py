#!/usr/bin/env python3
"""
Simple test to verify latency monitoring integration in chat_router.py
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

# Test that the metadata field is properly defined
try:
    from chat_router import ChatCompletionResponse

    print("✓ ChatCompletionResponse imported successfully")

    # Test that metadata field exists
    response = ChatCompletionResponse(
        id="test",
        model="test-model",
        provider="test-provider",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "test"}}],
        metadata={"response_time_ms": 100.0, "tokens_used": 10, "success": True},
    )
    print("✓ ChatCompletionResponse with metadata created successfully")
    print(f"✓ Metadata: {response.metadata}")

except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("✓ All tests passed!")
