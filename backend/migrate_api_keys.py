"""
Migration script to move API keys from file-based storage to encrypted storage.

This script migrates existing API keys from api_keys.json to the new encrypted API key store.
"""

import json
import sys
from pathlib import Path
from auth.api_key_store import APIKeyStore

# Add the backend directory to the path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))


def migrate_api_keys():
    """Migrate API keys from file-based storage to encrypted storage."""
    print("🔄 Starting API key migration...")

    # Paths
    old_keys_file = backend_dir / "api_keys.json"
    new_keys_file = backend_dir / "api_keys.enc"

    # Check if old file exists
    if not old_keys_file.exists():
        print("✅ No old API keys file found. Nothing to migrate.")
        return

    # Check if new file already exists
    if new_keys_file.exists():
        print("⚠️  New encrypted API keys file already exists.")
        response = input("Do you want to continue and potentially overwrite? (y/N): ")
        if response.lower() != "y":
            print("Migration cancelled.")
            return

    # Load old keys
    try:
        with open(old_keys_file, "r") as f:
            old_keys = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load old API keys: {e}")
        return

    if not old_keys:
        print("✅ Old API keys file is empty. Nothing to migrate.")
        return

    print(f"📋 Found {len(old_keys)} API keys to migrate")

    # Initialize new API key store
    try:
        api_key_store = APIKeyStore(str(new_keys_file))
    except Exception as e:
        print(f"❌ Failed to initialize API key store: {e}")
        return

    # Migrate each key
    migrated_count = 0
    failed_count = 0

    for provider, key in old_keys.items():
        try:
            print(f"  Migrating {provider}...")

            # Create new API key with provider-specific metadata
            key_id, _ = api_key_store.create_key(
                name=f"provider-{provider}",
                scopes=["read:models", "write:conversations"],  # Basic provider scopes
                metadata={
                    "provider": provider,
                    "migrated_from": "file_storage",
                    "original_key_hash": hash(key),  # For verification
                },
            )

            migrated_count += 1
            print(f"    ✅ Migrated {provider} -> {key_id}")

        except Exception as e:
            print(f"    ❌ Failed to migrate {provider}: {e}")
            failed_count += 1

    # Summary
    print("\nMigration Summary:")
    print(f"  Successfully migrated: {migrated_count}")
    print(f"  Failed to migrate: {failed_count}")
    print(f"  New encrypted keys stored in: {new_keys_file}")

    if migrated_count > 0:
        print(
            "\nIMPORTANT: The old API keys file still contains the actual key values!"
        )
        print("   After verifying the migration worked, you should:")
        print("   1. Test that the new encrypted keys work")
        print(f"   2. Remove the old file: rm {old_keys_file}")
        print("   3. Update any code that was reading from the old file")

    # Verify migration
    print("\nVerifying migration...")
    try:
        new_keys = api_key_store.list_keys()
        print(f"  Found {len(new_keys)} keys in encrypted store")

        # Check that we have the expected number
        if len(new_keys) == migrated_count:
            print("  Key count matches migrated count")
        else:
            print(
                f"  Key count mismatch: expected {migrated_count}, got {len(new_keys)}"
            )

    except Exception as e:
        print(f"  Failed to verify migration: {e}")

    print("\nMigration completed!")


if __name__ == "__main__":
    migrate_api_keys()
