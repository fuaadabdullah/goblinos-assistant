# pyright: reportMissingImports=false
#!/usr/bin/env python3
"""
Direct refactoring validation script - tests services directly
"""

import importlib
import sys
from pathlib import Path

# Add the backend directory to the path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))


def test_direct_imports():
    """Test that all services can be imported directly."""
    print("Testing direct imports...")

    try:
        # Test core services with direct imports
        importlib.import_module("services.chat_validator")
        importlib.import_module("services.chat_response_builder")
        importlib.import_module("services.chat_error_handler")
        importlib.import_module("services.chat_rate_limiter")
        importlib.import_module("services.chat_session_manager")
        importlib.import_module("services.chat_provider_selector")
        importlib.import_module("services.chat_metrics_collector")
        importlib.import_module("services.chat_cache_manager")
        importlib.import_module("services.chat_timeout_handler")
        importlib.import_module("services.chat_retry_handler")
        importlib.import_module("services.chat_compression_handler")
        importlib.import_module("services.chat_response_formatter")
        importlib.import_module("services.chat_error_formatter")
        importlib.import_module("services.chat_controller_refactored")

        print("✓ All service modules imported successfully")
        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_direct_instantiation():
    """Test that all services can be instantiated directly."""
    print("Testing direct instantiation...")

    try:
        # Import and instantiate services directly
        chat_validator = importlib.import_module("services.chat_validator")
        chat_response_builder = importlib.import_module("services.chat_response_builder")
        chat_error_handler = importlib.import_module("services.chat_error_handler")
        chat_rate_limiter = importlib.import_module("services.chat_rate_limiter")
        chat_session_manager = importlib.import_module("services.chat_session_manager")
        chat_provider_selector = importlib.import_module("services.chat_provider_selector")
        chat_metrics_collector = importlib.import_module("services.chat_metrics_collector")
        chat_cache_manager = importlib.import_module("services.chat_cache_manager")
        chat_timeout_handler = importlib.import_module("services.chat_timeout_handler")
        chat_retry_handler = importlib.import_module("services.chat_retry_handler")
        chat_compression_handler = importlib.import_module("services.chat_compression_handler")
        chat_response_formatter = importlib.import_module("services.chat_response_formatter")
        chat_error_formatter = importlib.import_module("services.chat_error_formatter")
        chat_controller_refactored = importlib.import_module("services.chat_controller_refactored")

        # Test service instantiation
        validator = chat_validator.ChatValidator()
        builder = chat_response_builder.ChatResponseBuilder()
        error_handler = chat_error_handler.ChatErrorHandler()
        rate_limiter = chat_rate_limiter.ChatRateLimiter()
        session_manager = chat_session_manager.ChatSessionManager()
        provider_selector = chat_provider_selector.ChatProviderSelector([])
        metrics_collector = chat_metrics_collector.ChatMetricsCollector()
        cache_manager = chat_cache_manager.ChatCacheManager()
        timeout_handler = chat_timeout_handler.ChatTimeoutHandler()
        retry_handler = chat_retry_handler.ChatRetryHandler()
        compression_handler = chat_compression_handler.ChatCompressionHandler()
        response_formatter = chat_response_formatter.ChatResponseFormatter()
        error_formatter = chat_error_formatter.ChatErrorFormatter()
        controller = chat_controller_refactored.ChatController()

        print("✓ All services instantiated successfully")
        return True

    except Exception as e:
        print(f"✗ Instantiation error: {e}")
        return False


def test_controller_services():
    """Test that the controller has all expected services."""
    print("Testing controller integration...")

    try:
        chat_controller_refactored = importlib.import_module("services.chat_controller_refactored")

        # Create controller
        controller = chat_controller_refactored.ChatController()

        # Verify controller has all expected services
        services_list = [
            "validator",
            "rate_limiter",
            "session_manager",
            "provider_selector",
            "response_builder",
            "error_handler",
            "metrics_collector",
            "cache_manager",
            "timeout_handler",
            "retry_handler",
            "compression_handler",
            "response_formatter",
            "error_formatter",
        ]

        for service in services_list:
            if not hasattr(controller, service):
                print(f"✗ Controller missing service: {service}")
                return False

        print("✓ Controller has all expected services")
        return True

    except Exception as e:
        print(f"✗ Controller test error: {e}")
        return False


def test_cache_direct():
    """Test basic cache functionality directly."""
    print("Testing cache functionality...")

    try:
        chat_cache_manager = importlib.import_module("services.chat_cache_manager")

        # Test cache manager
        cache_manager = chat_cache_manager.ChatCacheManager()
        cache_manager.set("test_key", {"test": "data"}, cache_manager.CacheType.RESPONSE)
        retrieved = cache_manager.get("test_key", cache_manager.CacheType.RESPONSE)

        if retrieved == {"test": "data"}:
            print("✓ Cache functionality works")
            return True
        else:
            print("✗ Cache functionality failed")
            return False

    except Exception as e:
        print(f"✗ Cache test error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🧪 Running Direct Refactoring Validation Tests")
    print("=" * 50)

    tests = [
        test_direct_imports,
        test_direct_instantiation,
        test_controller_services,
        test_cache_direct,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED - REFACTORING SUCCESSFUL!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
