#!/usr/bin/env python3
"""
Complete Database Connection Fix - CLI Only
"""
import os
import sys
import subprocess
import re
from pathlib import Path

def run_command(cmd, capture_output=False):
    """Run a shell command"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def update_env_file(new_password):
    """Update the .env file with the new password"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    content = env_file.read_text()
    
    # Replace the DATABASE_URL
    pattern = r'(DATABASE_URL=postgresql://postgres\.dhxoowakvmobjxsffpst:)[^@]+(@aws-0-us-west-2\.pooler\.supabase\.com:6543/postgres)'
    replacement = f'\\1{new_password}\\2'
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        env_file.write_text(new_content)
        print("✅ Updated .env file")
        return True
    else:
        print("❌ Could not find DATABASE_URL pattern in .env file")
        return False

def test_connection(password):
    """Test the database connection"""
    try:
        from sqlalchemy import create_engine, text
        url = f'postgresql://postgres.dhxoowakvmobjxsffpst:{password}@aws-0-us-west-2.pooler.supabase.com:6543/postgres'
        engine = create_engine(url)
        with engine.connect() as conn:
            result = conn.execute(text('SELECT version()'))
            version = result.fetchone()[0]
            print("✅ Database connection successful!")
            print(f"PostgreSQL version: {version.split()[1]}")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)[:150]}...")
        return False

def main():
    print("🔧 Complete Database Connection Fix")
    print("=" * 50)
    print()
    print("This script will help you fix the database connection issue.")
    print()
    
    # Step 1: Open browser
    print("📱 Step 1: Opening Supabase Dashboard")
    print("   Go to: https://supabase.com/dashboard/project/dhxoowakvmobjxsffpst/settings/database")
    run_command("open 'https://supabase.com/dashboard/project/dhxoowakvmobjxsffpst/settings/database'")
    print("   ✅ Browser opened")
    print()
    
    # Step 2: Guide user to reset password
    print("🔑 Step 2: Reset Database Password")
    print("   1. Click 'Reset database password'")
    print("   2. Confirm the reset")
    print("   3. Wait for the new password to appear")
    print("   4. Copy the new password")
    print()
    
    # Step 3: Get new password
    while True:
        new_password = input("Enter the new database password: ").strip()
        if not new_password:
            print("❌ Password cannot be empty")
            continue
        if len(new_password) < 8:
            print("❌ Password seems too short")
            continue
        break
    
    print(f"✅ Password received ({len(new_password)} characters)")
    print()
    
    # Step 4: Update configuration
    print("⚙️  Step 4: Updating Configuration")
    if not update_env_file(new_password):
        print("❌ Failed to update configuration")
        return False
    
    # Step 5: Test connection
    print("🧪 Step 5: Testing Connection")
    if test_connection(new_password):
        print()
        print("🎉 SUCCESS! Database connection is now working!")
        print("   The Goblin Assistant backend should now be able to connect to the database.")
        return True
    else:
        print()
        print("❌ Connection test failed. The password might be incorrect.")
        print("   Please try again or check the Supabase dashboard.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
