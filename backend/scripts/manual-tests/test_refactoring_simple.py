# pyright: reportMissingImports=false
#!/usr/bin/env python3
"""
Simple refactoring validation script
"""

import importlib
import sys
from pathlib import Path

# Add the backend directory to the path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))


def test_imports():
    """Test that all services can be imported."""
    print("Testing imports...")

    try:
        # Test core services
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

        print("✓ All service imports successful")
        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_instantiation():
    """Test that all services can be instantiated."""
    print("Testing instantiation...")

    try:
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


def test_controller():
    """Test that the controller has all expected services."""
    print("Testing controller integration...")

    try:
        chat_controller_refactored = importlib.import_module("services.chat_controller_refactored")

        # Create controller
        controller = chat_controller_refactored.ChatController()

        # Verify controller has all expected services
        services = [
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

        for service in services:
            if not hasattr(controller, service):
                print(f"✗ Controller missing service: {service}")
                return False

        print("✓ Controller has all expected services")
        return True

    except Exception as e:
        print(f"✗ Controller test error: {e}")
        return False


def test_cache_functionality():
    """Test basic cache functionality."""
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
    print("🧪 Running Simple Refactoring Validation Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_instantiation,
        test_controller,
        test_cache_functionality,
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
