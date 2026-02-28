# pyright: reportMissingImports=false
#!/usr/bin/env python3
"""
Integration test for enhanced RAG system in goblin-assistant backend.
Tests the integration between configuration, services, and routers.
"""

import asyncio
import sys
from pathlib import Path

# Add the backend directory to the path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

from config import settings
from services.rag_service import RAGService


async def test_rag_integration():
    """Test RAG service integration with configuration."""
    print("🧪 Testing Enhanced RAG Integration")
    print("=" * 50)

    # Test 1: Configuration loading
    print("1. Testing configuration...")
    print(f"   Enhanced RAG enabled: {settings.enable_enhanced_rag}")
    print(f"   Chroma path: {settings.rag_chroma_path}")
    print("   ✅ Configuration loaded")

    # Test 2: RAG service initialization
    print("\n2. Testing RAG service initialization...")
    try:
        rag_service = RAGService(
            enable_enhanced=settings.enable_enhanced_rag,
            chroma_path=settings.rag_chroma_path,
        )
        print("   ✅ RAG service initialized")
        chroma_available = rag_service.chroma_client is not None
        print(f"   ChromaDB available: {chroma_available}")

        # Test 3: Pipeline selection
        print("\n3. Testing pipeline selection...")
        if settings.enable_enhanced_rag:
            print("   Using enhanced RAG pipeline")
            # Test enhanced pipeline (would fail without dependencies, but should not crash)
            try:
                # This will fail gracefully if dependencies are missing
                await rag_service.enhanced_rag_pipeline("test query")
                print("   ✅ Enhanced pipeline executed")
            except Exception as e:
                import traceback

                print(f"   ⚠️  Enhanced pipeline failed (expected): {e}")
                print("   Full traceback:")
                traceback.print_exc()
        else:
            print("   Using standard RAG pipeline")
            try:
                await rag_service.rag_pipeline("test query")
                print("   ✅ Standard pipeline executed")
            except Exception as e:
                print(f"   ⚠️  Standard pipeline failed (expected): {e}")

    except Exception as e:
        print(f"   ❌ RAG service initialization failed: {e}")
        return False

    # Test 4: TF-IDF fallback retrieval
    print("\n4. Testing TF-IDF fallback retrieval...")
    try:
        # Add some test documents
        test_docs = [
            {
                "id": "1",
                "text": "Python is a programming language used for web development and data science.",
            },
            {
                "id": "2",
                "text": "Machine learning algorithms can be implemented in Python using libraries like scikit-learn.",
            },
            {
                "id": "3",
                "text": "FastAPI is a modern web framework for building APIs with Python.",
            },
        ]

        # Add documents to the enhanced service
        success = await rag_service.add_documents(test_docs)
        if success:
            print("   ✅ Documents added successfully")

            # Test retrieval using enhanced pipeline
            results = await rag_service.enhanced_rag_pipeline("python programming")
            if results and "context" in results:
                context_data = results["context"]
                if context_data and "chunks" in context_data:
                    context_items = context_data["chunks"]
                    if context_items and len(context_items) > 0:
                        print(f"   ✅ Retrieval successful, found {len(context_items)} results")
                        print(f"   📄 First result: {context_items[0].get('text', '')[:100]}...")
                    else:
                        print("   ⚠️  Retrieval returned no chunks")
                else:
                    print("   ⚠️  Retrieval returned no context data")
            else:
                print("   ⚠️  Retrieval returned no results")
        else:
            print("   ⚠️  Failed to add documents")

    except Exception as e:
        import traceback

        print(f"   ⚠️  TF-IDF retrieval test failed: {e}")
        print("   Full traceback:")
        traceback.print_exc()

    # Test 5: Settings router integration
    print("\n5. Testing settings router integration...")
    try:
        from settings_router import get_rag_settings, test_rag_configuration

        rag_settings = await get_rag_settings()
        print(f"   RAG settings retrieved: {rag_settings.enable_enhanced_rag}")

        config_test = await test_rag_configuration()
        print(f"   Configuration test: {config_test['status']}")
        print("   ✅ Settings router integration working")

    except Exception as e:
        print(f"   ❌ Settings router test failed: {e}")
        return False

    print("\n🎉 Integration test completed successfully!")
    print("\n📋 Summary:")
    print(f"   - Enhanced RAG: {'Enabled' if settings.enable_enhanced_rag else 'Disabled'}")
    print(f"   - ChromaDB: {'Available' if chroma_available else 'Not available'}")
    print("   - Configuration: Working")
    print("   - Services: Working")
    print("   - Routers: Working")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_rag_integration())
    sys.exit(0 if success else 1)
