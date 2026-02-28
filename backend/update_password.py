#!/usr/bin/env python3
import os
import sys

def update_password():
    print("🔄 Database Password Update Tool")
    print("=" * 40)
    
    # Get new password from user
    new_password = input("Enter the new database password from Supabase: ").strip()
    
    if not new_password:
        print("❌ No password provided")
        return False
    
    # Update .env file
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Replace the DATABASE_URL line
        import re
        pattern = r'(DATABASE_URL=postgresql://postgres\.dhxoowakvmobjxsffpst:)[^@]+(@aws-0-us-west-2\.pooler\.supabase\.com:6543/postgres)'
        new_url = f'postgresql://postgres.dhxoowakvmobjxsffpst:{new_password}@aws-0-us-west-2.pooler.supabase.com:6543/postgres'
        replacement = f'\\1{new_password}\\2'
        
        if re.search(pattern, content):
            new_content = re.sub(pattern, replacement, content)
            with open(env_file, 'w') as f:
                f.write(new_content)
            print("✅ Updated .env file")
        else:
            print("❌ Could not find DATABASE_URL in .env file")
            return False
    else:
        print("❌ .env file not found")
        return False
    
    # Test connection
    print("\n🧪 Testing database connection...")
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(f'postgresql://postgres.dhxoowakvmobjxsffpst:{new_password}@aws-0-us-west-2.pooler.supabase.com:6543/postgres')
        with engine.connect() as conn:
            result = conn.execute(text('SELECT 1'))
            print("✅ Database connection successful!")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)[:100]}...")
        return False

if __name__ == "__main__":
    success = update_password()
    sys.exit(0 if success else 1)
