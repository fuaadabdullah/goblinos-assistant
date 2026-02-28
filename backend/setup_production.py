#!/usr/bin/env python3
"""
Production Database Setup
Updates .env file with PostgreSQL and Redis configuration

Usage:
  python3 setup_production.py --db-url "postgresql://..."
  python3 setup_production.py --db-url "postgresql://..." --skip-redis
  python3 setup_production.py --db-url "postgresql://..." --redis-host "host" --redis-port "6379" --redis-password "pass" --redis-ssl true
"""

import os
import sys
from pathlib import Path
import shutil
from datetime import datetime
import argparse


def main():
    parser = argparse.ArgumentParser(description="Production Database Setup")
    parser.add_argument("--db-url", required=True, help="PostgreSQL connection string")
    parser.add_argument(
        "--skip-redis", action="store_true", help="Skip Redis configuration"
    )
    parser.add_argument("--redis-host", help="Redis host")
    parser.add_argument("--redis-port", default="6379", help="Redis port")
    parser.add_argument("--redis-password", help="Redis password")
    parser.add_argument("--redis-ssl", default="true", help="Use SSL for Redis")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    env_file = script_dir / ".env"

    if not env_file.exists():
        print("❌ .env file not found!")
        print(f"   Looking for: {env_file}")
        sys.exit(1)

    print("=" * 60)
    print("🚀 PRODUCTION DATABASE SETUP")
    print("=" * 60)
    print()

    # Create backup
    backup_file = (
        env_file.parent / f".env.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy(env_file, backup_file)
    print(f"✅ Created backup: {backup_file.name}")
    print()

    # Validate PostgreSQL URL
    db_url = args.db_url.strip()
    if not db_url.startswith("postgresql://"):
        print("❌ Invalid PostgreSQL URL format. Must start with 'postgresql://'")
        sys.exit(1)

    print("✅ PostgreSQL connection string accepted")
    print()

    # Handle Redis
    if args.skip_redis:
        print("⚠️  Skipping Redis (will use in-memory storage)")
        use_redis = "false"
        redis_host = "localhost"
        redis_port = "6379"
        redis_password = ""
        redis_ssl_value = "false"
    else:
        if not args.redis_host or not args.redis_password:
            print(
                "❌ Redis configuration incomplete. Provide --redis-host and --redis-password, or use --skip-redis"
            )
            sys.exit(1)

        use_redis = "true"
        redis_host = args.redis_host
        redis_port = args.redis_port
        redis_password = args.redis_password
        redis_ssl_value = args.redis_ssl

        print("✅ Redis configuration accepted")

    print()

    # Update .env file
    print("─" * 60)
    print("📝 STEP 3: Updating .env file")
    print("─" * 60)
    print()

    with open(env_file, "r") as f:
        content = f.read()

    # Update DATABASE_URL
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        if line.startswith("DATABASE_URL="):
            new_lines.append(f"DATABASE_URL={db_url}")
            print("✅ Updated DATABASE_URL")
        elif line.startswith("USE_REDIS_CHALLENGES="):
            new_lines.append(f"USE_REDIS_CHALLENGES={use_redis}")
            print("✅ Updated USE_REDIS_CHALLENGES")
        elif line.startswith("REDIS_HOST="):
            new_lines.append(f"REDIS_HOST={redis_host}")
            print("✅ Updated REDIS_HOST")
        elif line.startswith("REDIS_PORT="):
            new_lines.append(f"REDIS_PORT={redis_port}")
            print("✅ Updated REDIS_PORT")
        elif line.startswith("REDIS_PASSWORD="):
            new_lines.append(f"REDIS_PASSWORD={redis_password}")
            print("✅ Updated REDIS_PASSWORD")
        elif line.startswith("REDIS_SSL="):
            new_lines.append(f"REDIS_SSL={redis_ssl_value}")
            print("✅ Updated REDIS_SSL")
        else:
            new_lines.append(line)

    with open(env_file, "w") as f:
        f.write("\n".join(new_lines))

    print()
    print("=" * 60)
    print("✅ CONFIGURATION COMPLETE!")
    print("=" * 60)
    print()
    print("Next step: Test connection and run migrations")
    print()
    print("Run these commands:")
    print("  source venv/bin/activate")
    print("  python3 test_db_connection.py")
    print("  alembic upgrade head")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
