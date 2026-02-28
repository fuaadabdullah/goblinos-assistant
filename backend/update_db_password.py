#!/usr/bin/env python3
"""
Helper script to update Supabase database password in .env file
"""
import sys
import re
from pathlib import Path

def update_database_password(new_password):
    env_file = Path(__file__).parent / '.env'
    
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    content = env_file.read_text()
    
    # Pattern to match DATABASE_URL
    pattern = r'(DATABASE_URL=postgresql://postgres\.dhxoowakvmobjxsffpst:)([^@]+)(@aws-0-us-west-2\.pooler\.supabase\.com:6543/postgres)'
    
    # Replace password
    new_content = re.sub(pattern, rf'\g<1>{new_password}\g<3>', content)
    
    if new_content == content:
        print("⚠️  No changes made - pattern not found or password unchanged")
        return False
    
    # Backup old .env
    backup_file = env_file.parent / '.env.backup'
    backup_file.write_text(content)
    print(f"✅ Backed up current .env to {backup_file}")
    
    # Write new .env
    env_file.write_text(new_content)
    print("✅ Updated DATABASE_URL with new password")
    
    # Test connection
    print("\n🔍 Testing connection...")
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    try:
        import psycopg2
        db_url = os.getenv('DATABASE_URL')
        conn = psycopg2.connect(db_url)
        print("✅ CONNECTION SUCCESSFUL!")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nRestoring backup...")
        env_file.write_text(content)
        print("✅ Restored original .env")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_db_password.py <new_password>")
        print("\nGet your password from:")
        print("https://supabase.com/dashboard/project/dhxoowakvmobjxsffpst/settings/database")
        sys.exit(1)
    
    new_password = sys.argv[1]
    update_database_password(new_password)
