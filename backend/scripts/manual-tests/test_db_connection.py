# pyright: reportMissingImports=false
#!/usr/bin/env python3
"""
Test Database Connection
Verifies PostgreSQL and Redis connections
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

load_dotenv()

print("=" * 60)
print("🔍 TESTING DATABASE CONNECTIONS")
print("=" * 60)
print()

# Test PostgreSQL
print("─" * 60)
print("📊 PostgreSQL Connection")
print("─" * 60)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL not set in .env")
    sys.exit(1)

# Hide password in display
display_url = DATABASE_URL
if "@" in display_url:
    parts = display_url.split("@")
    if ":" in parts[0]:
        user_pass = parts[0].split("://")[-1]
        user = user_pass.split(":")[0]
        display_url = display_url.replace(user_pass, f"{user}:****")

print(f"Database: {display_url}")
print()

try:
    engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 10})

    with engine.connect() as conn:
        # Test connection
        result = conn.execute(text("SELECT version()"))
        version = result.scalar_one_or_none()
        if version is None:
            raise RuntimeError("Failed to fetch PostgreSQL version.")

        # Get database info
        result = conn.execute(text("SELECT current_database()"))
        db_name = result.scalar_one_or_none()
        if db_name is None:
            raise RuntimeError("Failed to fetch current database name.")

        print("✅ PostgreSQL Connection Successful!")
        print(f"   Database: {db_name}")
        print(f"   Host: {engine.url.host}")
        print(f"   Port: {engine.url.port}")
        print(f"   Version: {version[:80]}...")

except Exception as e:
    print("❌ PostgreSQL Connection Failed!")
    print(f"   Error: {e}")
    print()
    print("Troubleshooting:")
    print("  - Check your connection string in .env")
    print("  - Verify database password is correct")
    print("  - Ensure you're using port 6543 (connection pooling)")
    print("  - Check Supabase project is not paused")
    sys.exit(1)

print()

# Test Redis
print("─" * 60)
print("🔴 Redis Connection")
print("─" * 60)

use_redis = os.getenv("USE_REDIS_CHALLENGES", "false").lower() == "true"

if not use_redis:
    print("⚠️  Redis not configured (using in-memory storage)")
    print("   To enable: Set USE_REDIS_CHALLENGES=true in .env")
else:
    try:
        import redis

        redis_host = os.getenv("REDIS_HOST")
        if not redis_host:
            print("❌ REDIS_HOST not set in .env")
            sys.exit(1)
        assert redis_host is not None
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD")
        redis_ssl = os.getenv("REDIS_SSL", "false").lower() == "true"

        print(f"Host: {redis_host}")
        print(f"Port: {redis_port}")
        print(f"SSL: {redis_ssl}")
        print()

        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            ssl=redis_ssl,
            decode_responses=True,
        )

        # Test connection
        r.ping()

        # Test set/get
        test_key = "test:connection"
        r.setex(test_key, 10, "test_value")
        value = r.get(test_key)
        r.delete(test_key)

        print("✅ Redis Connection Successful!")
        print(f"   Connected to {redis_host}")
        print("   Test write/read: OK")

    except ImportError:
        print("❌ Redis library not installed")
        print("   Run: pip install redis")
        sys.exit(1)
    except Exception as e:
        print("❌ Redis Connection Failed!")
        print(f"   Error: {e}")
        print()
        print("Troubleshooting:")
        print("  - Check REDIS_HOST, REDIS_PASSWORD in .env")
        print("  - Verify REDIS_SSL=true for Upstash")
        print("  - Test with: redis-cli -h <host> -a <password> --tls PING")
        sys.exit(1)

print()
print("=" * 60)
print("✅ ALL CONNECTIONS SUCCESSFUL!")
print("=" * 60)
print()
print("Next step: Run migrations")
print("  alembic upgrade head")
print()
