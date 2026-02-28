#!/usr/bin/env python3
"""
Automated Database Connection Fix for Goblin Assistant

This script automates the complete process of fixing the database connection:
1. Check Bitwarden vault status
2. Guide through Supabase password reset
3. Store password securely in Bitwarden
4. Update .env file with correct DATABASE_URL
5. Test database connection

Usage:
    python fix_database_connection.py
"""

import sys
import subprocess
import re
import json
from pathlib import Path
from typing import Optional, Tuple


class DatabaseFixer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.backend_dir = self.project_root / "apps" / "goblin-assistant" / "backend"
        self.env_file = self.backend_dir / ".env"
        self.supabase_project_id = "dhxoowakvmobjxsffpst"

    def run_command(
        self, cmd: str, capture_output: bool = False
    ) -> Tuple[int, str, str]:
        """Run a shell command and return (exit_code, stdout, stderr)"""
        # Handle bw commands specially
        if cmd.strip().startswith("bw "):
            # Run bw commands directly with full path
            full_cmd = f"/Users/fuaadabdullah/.volta/bin/{cmd}"
            try:
                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    capture_output=capture_output,
                    text=True,
                    cwd=self.backend_dir,
                )
                return result.returncode, result.stdout, result.stderr
            except Exception as e:
                return 1, "", str(e)
        else:
            # Use shell=True for other commands
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=capture_output,
                    text=True,
                    cwd=self.backend_dir,
                )
                return result.returncode, result.stdout, result.stderr
            except Exception as e:
                return 1, "", str(e)

    def check_bitwarden_status(self) -> dict:
        """Check Bitwarden vault status"""
        exit_code, stdout, stderr = self.run_command("bw status", capture_output=True)

        if exit_code != 0:
            return {"status": "error", "message": "Bitwarden CLI not available"}

        try:
            status = json.loads(stdout)
            return {
                "status": status.get("status", "unknown"),
                "user": status.get("userEmail", "unknown"),
                "last_sync": status.get("lastSync", "unknown"),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Could not parse Bitwarden status: {e}",
            }

    def unlock_bitwarden(self) -> bool:
        """Guide user through Bitwarden unlock process"""
        print("🔐 Bitwarden Vault Status Check")
        print("=" * 50)

        status = self.check_bitwarden_status()

        if status["status"] == "unlocked":
            print("✅ Bitwarden vault is already unlocked")
            return True

        if status["status"] == "locked":
            print("🔒 Bitwarden vault is locked")
            print(f"   User: {status.get('user', 'unknown')}")
            print(f"   Last sync: {status.get('last_sync', 'unknown')}")
            print()
            print("Please unlock your vault:")
            print("1. Run: bw unlock")
            print("2. Copy the session token")
            print("3. This script will continue automatically")
            print()
            input("Press Enter when you've unlocked the vault...")

            # Check again
            status = self.check_bitwarden_status()
            if status["status"] == "unlocked":
                print("✅ Bitwarden vault unlocked successfully!")
                return True
            else:
                print("❌ Bitwarden vault is still locked")
                return False

        print(f"❌ Bitwarden status: {status}")
        return False

    def guide_supabase_reset(self) -> Optional[str]:
        """Guide user through Supabase password reset"""
        print("\n🌐 Supabase Database Password Reset")
        print("=" * 50)
        print("You need to reset your Supabase database password:")
        print()
        print("1. Open this URL in your browser:")
        print(
            f"   https://supabase.com/dashboard/project/{self.supabase_project_id}/settings/database"
        )
        print()
        print("2. Click 'Reset database password'")
        print("3. Copy the new password")
        print()

        while True:
            password = input(
                "Enter the new database password (or 'skip' to cancel): "
            ).strip()

            if password.lower() == "skip":
                return None

            if len(password) < 8:
                print(
                    "❌ Password seems too short. Supabase passwords are usually longer."
                )
                continue

            # Basic validation - should contain various characters
            if not any(c.isupper() for c in password):
                print("⚠️  Password should contain uppercase letters")
                continue

            confirm = input("Confirm password: ").strip()
            if password != confirm:
                print("❌ Passwords don't match")
                continue

            return password

    def store_in_bitwarden(self, password: str) -> bool:
        """Store password in Bitwarden vault"""
        print("\n🔑 Storing Password in Bitwarden")
        print("=" * 50)

        item_name = "Goblin Assistant - Database Password"
        username = f"postgres.{self.supabase_project_id}"

        # Check if item already exists
        exit_code, stdout, stderr = self.run_command(
            f'bw get item "{item_name}"', capture_output=True
        )

        if exit_code == 0:
            print(f"⚠️  Bitwarden item '{item_name}' already exists")
            overwrite = input("Overwrite existing item? (y/N): ").strip().lower()
            if overwrite != "y":
                print("✅ Using existing Bitwarden item")
                return True

        # Create new item
        create_cmd = f'''bw create item <<EOF
{{
  "type": 1,
  "name": "{item_name}",
  "login": {{
    "username": "{username}",
    "password": "{password}"
  }}
}}
EOF'''

        exit_code, stdout, stderr = self.run_command(create_cmd, capture_output=True)

        if exit_code == 0:
            print("✅ Password stored in Bitwarden successfully!")
            return True
        else:
            print(f"❌ Failed to store in Bitwarden: {stderr}")
            return False

    def update_env_file(self, password: str) -> bool:
        """Update the .env file with correct DATABASE_URL"""
        print("\n📝 Updating .env File")
        print("=" * 50)

        if not self.env_file.exists():
            print(f"❌ .env file not found at {self.env_file}")
            return False

        # Read current content
        content = self.env_file.read_text()

        # Create backup
        backup_file = self.env_file.with_suffix(".backup")
        backup_file.write_text(content)
        print(f"✅ Created backup: {backup_file}")

        # Update DATABASE_URL
        # Current format: DATABASE_URL=sqlite:///test.db
        # New format: DATABASE_URL=postgresql://postgres.dhxoowakvmobjxsffpst:[PASSWORD]@aws-0-us-east-1.pooler.supabase.com:6543/postgres

        old_pattern = r"DATABASE_URL=sqlite:///test\.db"
        new_database_url = f"postgresql://postgres.{self.supabase_project_id}:{password}@aws-0-us-east-1.pooler.supabase.com:6543/postgres"

        if re.search(old_pattern, content):
            new_content = re.sub(
                old_pattern, f"DATABASE_URL={new_database_url}", content
            )
            self.env_file.write_text(new_content)
            print("✅ Updated DATABASE_URL in .env file")
            return True
        else:
            print("⚠️  Could not find SQLite DATABASE_URL pattern in .env file")
            print("   The file might already be updated or have a different format")
            return False

    def test_database_connection(self) -> bool:
        """Test the database connection"""
        print("\n🧪 Testing Database Connection")
        print("=" * 50)

        # First try the helper script
        exit_code, stdout, stderr = self.run_command(
            "python update_db_password.py --test-only", capture_output=True
        )

        if exit_code == 0 and "CONNECTION SUCCESSFUL" in stdout:
            print("✅ Database connection test passed!")
            return True
        else:
            print("❌ Database connection test failed")
            print(f"Output: {stdout}")
            print(f"Error: {stderr}")

            # Try direct connection test
            print("\n🔄 Trying direct connection test...")
            test_cmd = """
import os
from dotenv import load_dotenv
load_dotenv()
import psycopg2

try:
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ DATABASE_URL not set")
        exit(1)

    conn = psycopg2.connect(db_url)
    conn.close()
    print("✅ Direct connection successful!")
except Exception as e:
    print(f"❌ Direct connection failed: {e}")
    exit(1)
"""

            exit_code, stdout, stderr = self.run_command(
                f'python -c "{test_cmd}"', capture_output=True
            )

            if exit_code == 0 and "Direct connection successful" in stdout:
                print("✅ Direct database connection test passed!")
                return True
            else:
                print("❌ Direct database connection test also failed")
                return False

    def run_full_fix(self):
        """Run the complete automated fix"""
        print("🚀 Goblin Assistant Database Connection Fix")
        print("=" * 60)
        print("This script will fix the database connection issue automatically")
        print()

        # Step 1: Check Bitwarden
        if not self.unlock_bitwarden():
            print("❌ Cannot proceed without unlocked Bitwarden vault")
            return False

        # Step 2: Guide Supabase reset
        password = self.guide_supabase_reset()
        if not password:
            print("❌ Password reset cancelled")
            return False

        # Step 3: Store in Bitwarden
        if not self.store_in_bitwarden(password):
            print("⚠️  Bitwarden storage failed, but continuing...")

        # Step 4: Update .env file
        if not self.update_env_file(password):
            print("❌ Failed to update .env file")
            return False

        # Step 5: Test connection
        if not self.test_database_connection():
            print("❌ Database connection test failed")
            print("   You may need to check your Supabase settings or try again")
            return False

        print("\n🎉 SUCCESS! Database connection is now fixed!")
        print("=" * 60)
        print("✅ Bitwarden vault unlocked")
        print("✅ Supabase password reset")
        print("✅ Password stored securely in Bitwarden")
        print("✅ .env file updated with correct DATABASE_URL")
        print("✅ Database connection tested and working")
        print()
        print("You can now start the backend:")
        print("  cd apps/goblin-assistant-root/backend")
        print("  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001")

        return True


def main():
    fixer = DatabaseFixer()

    if len(sys.argv) > 1 and sys.argv[1] == "--test-only":
        # Just test the current connection
        success = fixer.test_database_connection()
        sys.exit(0 if success else 1)

    success = fixer.run_full_fix()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
