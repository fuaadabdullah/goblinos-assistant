# pyright: reportMissingImports=false
"""
Test script for ElevenLabs text-to-speech adapter.
Tests voice synthesis, voice listing, and streaming capabilities.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))


# Load environment variables from .env file
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


load_env_direct()

from providers.elevenlabs_adapter import ElevenLabsAdapter


async def test_health_check():
    """Test ElevenLabs API health check."""
    print("\n" + "=" * 60)
    print("Testing ElevenLabs Health Check")
    print("=" * 60)

    try:
        adapter = ElevenLabsAdapter()
        is_healthy = await adapter.health_check()

        if is_healthy:
            print("✅ Health Check: PASSED")
        else:
            print("❌ Health Check: FAILED")

        return is_healthy
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False


async def test_list_voices():
    """Test listing available voices."""
    print("\n" + "=" * 60)
    print("Testing Voice Listing")
    print("=" * 60)

    try:
        adapter = ElevenLabsAdapter()
        voices_data = await adapter.list_voices()

        voices = voices_data.get("voices", [])
        print(f"✅ Found {len(voices)} voices")

        # Display first 5 voices
        print("\nAvailable Voices (first 5):")
        for voice in voices[:5]:
            voice_id = voice.get("voice_id", "unknown")
            name = voice.get("name", "Unknown")
            category = voice.get("category", "unknown")
            labels = voice.get("labels", {})

            print(f"  - {name} ({voice_id})")
            print(f"    Category: {category}")
            if labels:
                print(f"    Labels: {labels}")

        return True
    except Exception as e:
        print(f"❌ Voice Listing Error: {e}")
        return False


async def test_voice_details():
    """Test getting details for specific voice."""
    print("\n" + "=" * 60)
    print("Testing Voice Details")
    print("=" * 60)

    try:
        adapter = ElevenLabsAdapter()
        voice_id = "JBFqnCBsd6RMkjVDRZzb"  # George voice

        details = await adapter.get_voice_details(voice_id)

        print(f"✅ Voice Details for {voice_id}:")
        print(f"  Name: {details.get('name', 'Unknown')}")
        print(f"  Category: {details.get('category', 'unknown')}")
        print(f"  Description: {details.get('description', 'No description')}")

        labels = details.get("labels", {})
        if labels:
            print(f"  Labels: {labels}")

        return True
    except Exception as e:
        print(f"❌ Voice Details Error: {e}")
        return False


async def test_generate_speech():
    """Test basic speech generation."""
    print("\n" + "=" * 60)
    print("Testing Speech Generation")
    print("=" * 60)

    try:
        adapter = ElevenLabsAdapter()

        messages = [
            {
                "role": "user",
                "content": "The first move is what sets everything in motion.",
            }
        ]

        print("Generating speech...")
        import time

        start = time.time()

        result = await adapter.generate(
            messages,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # George voice
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        elapsed = (time.time() - start) * 1000

        print(f"✅ Speech Generated:")
        print(f"  Duration: {elapsed:.2f}ms")
        print(f"  Size: {result['size_bytes']:,} bytes ({result['size_bytes'] / 1024:.2f} KB)")
        print(f"  Format: {result['format']}")
        print(f"  Voice ID: {result['voice_id']}")
        print(f"  Model: {result['model_id']}")
        print(f"  Text: {result['text']}")

        # Save audio file
        output_path = BACKEND_DIR / "test_output.mp3"
        with open(output_path, "wb") as f:
            f.write(result["audio"])

        print(f"\n💾 Audio saved to: {output_path}")
        print(f"   You can play it with: open {output_path}")

        return True
    except Exception as e:
        print(f"❌ Speech Generation Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_streaming_speech():
    """Test streaming speech generation."""
    print("\n" + "=" * 60)
    print("Testing Streaming Speech Generation")
    print("=" * 60)

    try:
        adapter = ElevenLabsAdapter()

        messages = [{"role": "user", "content": "This is a test of streaming audio generation."}]

        print("Streaming speech...")
        import time

        start = time.time()

        chunks = []
        chunk_count = 0

        async for chunk in adapter.stream_generate(
            messages, voice_id="JBFqnCBsd6RMkjVDRZzb", model_id="eleven_multilingual_v2"
        ):
            chunks.append(chunk)
            chunk_count += 1

        elapsed = (time.time() - start) * 1000
        total_bytes = sum(len(chunk) for chunk in chunks)

        print(f"✅ Streaming Complete:")
        print(f"  Duration: {elapsed:.2f}ms")
        print(f"  Chunks: {chunk_count}")
        print(f"  Total Size: {total_bytes:,} bytes ({total_bytes / 1024:.2f} KB)")

        # Save streamed audio
        output_path = BACKEND_DIR / "test_stream_output.mp3"
        with open(output_path, "wb") as f:
            for chunk in chunks:
                f.write(chunk)

        print(f"\n💾 Streamed audio saved to: {output_path}")

        return True
    except Exception as e:
        print(f"❌ Streaming Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_capabilities():
    """Test getting adapter capabilities."""
    print("\n" + "=" * 60)
    print("Testing Adapter Capabilities")
    print("=" * 60)

    try:
        adapter = ElevenLabsAdapter()
        capabilities = adapter.get_capabilities()

        print("✅ Adapter Capabilities:")
        print(f"  Provider: {capabilities['provider']}")
        print(f"  Type: {capabilities['type']}")
        print(f"  Streaming: {capabilities['streaming']}")
        print(f"  Voice Cloning: {capabilities['voice_cloning']}")
        print(f"  Multi-Language: {capabilities['multi_language']}")
        print(f"  Output Formats: {len(capabilities['output_formats'])} available")
        print(f"  Models: {len(capabilities['models'])} available")

        return True
    except Exception as e:
        print(f"❌ Capabilities Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ELEVENLABS TEXT-TO-SPEECH ADAPTER TEST SUITE")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ ERROR: ELEVENLABS_API_KEY not found in environment")
        print("Please set the API key in backend/.env file")
        return

    print(f"✅ API Key loaded: {len(api_key)} characters")

    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Capabilities", test_capabilities),
        ("List Voices", test_list_voices),
        ("Voice Details", test_voice_details),
        ("Generate Speech", test_generate_speech),
        ("Streaming Speech", test_streaming_speech),
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
        print("\n🎉 All tests passed! ElevenLabs integration is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
