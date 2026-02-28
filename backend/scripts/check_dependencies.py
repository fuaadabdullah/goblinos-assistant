#!/usr/bin/env python3
"""
Dependency verification script for Goblin Assistant backend.

Checks that all required dependencies are installed and properly configured.
Run this before deployment to catch dependency issues early.
"""

import sys
import importlib


def check_dependency(name: str, package_name: str = None, extras: str = None) -> bool:
    """Check if a dependency is available"""
    if package_name is None:
        package_name = name

    try:
        importlib.import_module(package_name)
        print(f"✅ {name} installed")
        return True
    except ImportError:
        if extras:
            print(
                f"❌ {name} missing - install with: pip install {package_name}[{extras}]"
            )
        else:
            print(f"❌ {name} missing - install with: pip install {package_name}")
        return False


def check_pydantic_email() -> bool:
    """Special check for pydantic[email] and email-validator"""
    try:
        from pydantic import BaseModel, EmailStr, ValidationError

        # Create a test model to validate EmailStr
        class TestEmail(BaseModel):
            email: EmailStr

        # Test valid email
        test_model = TestEmail(email="test@example.com")
        assert str(test_model.email) == "test@example.com"

        # Test invalid email validation
        try:
            TestEmail(email="invalid-email")
            return False  # Should have failed
        except ValidationError:
            pass  # Expected

        print("✅ pydantic[email] and email-validator installed and working")
        return True
    except ImportError as e:
        if "email_validator" in str(e):
            print(
                "❌ email-validator missing - install with: pip install email-validator"
            )
        else:
            print(
                "❌ pydantic[email] missing - install with: pip install pydantic[email]"
            )
        return False
    except Exception as e:
        print(f"❌ EmailStr validation failed: {e}")
        return False


def check_redis() -> bool:
    """Check Redis availability (optional for auth fallback)"""
    try:
        importlib.import_module("redis")

        print("✅ redis installed")
        return True
    except ImportError:
        print("⚠️  redis missing - auth fallback will be used (OK for single-instance)")
        return False


def check_database_connection() -> bool:
    """Check if database connection works"""
    try:
        # Add current directory to path for database import
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from database import get_db

        db = next(get_db())
        from sqlalchemy import text

        db.execute(text("SELECT 1"))
        db.close()
        print("✅ database connection working")
        return True
    except Exception as e:
        print(f"❌ database connection failed: {e}")
        return False


def main():
    """Main dependency check"""
    print("🔍 Checking Goblin Assistant Backend Dependencies")
    print("=" * 50)

    # Critical dependencies
    critical_deps = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("sqlalchemy", "sqlalchemy"),
        ("bcrypt", "bcrypt"),
        ("PyJWT", "jwt"),
    ]

    # Optional but recommended
    optional_deps = [
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("google-generativeai", "google.generativeai"),
    ]

    all_good = True

    # Check critical dependencies
    print("\n📦 Critical Dependencies:")
    for name, package in critical_deps:
        if not check_dependency(name, package):
            all_good = False

    # Special checks
    print("\n🔧 Special Requirements:")
    if not check_pydantic_email():
        all_good = False

    check_redis()  # Optional, don't fail on this

    # Check optional dependencies
    print("\n📦 Optional Dependencies:")
    for name, package in optional_deps:
        check_dependency(name, package)

    # Database check
    print("\n🗄️  Database:")
    if not check_database_connection():
        all_good = False

    print(f"\n{'=' * 50}")
    if all_good:
        print("🎉 All critical dependencies satisfied!")
        return 0
    else:
        print("❌ Some critical dependencies are missing or misconfigured")
        print("   Run: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
