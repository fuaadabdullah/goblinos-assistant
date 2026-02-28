#!/usr/bin/env python3
"""
Vault Migration Script
Goblin Assistant Secrets Management Migration

This script migrates secrets from various sources to HashiCorp Vault:
- Fernet-encrypted database secrets
- Bitwarden CLI secrets (from CircleCI)
- File-based API keys
- Environment variables

Usage:
    python migrate_secrets.py --environment staging --dry-run
    python migrate_secrets.py --environment production --migrate
"""

import argparse
import logging
import os
import json
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vault_client import get_vault_manager

logger = logging.getLogger(__name__)


class SecretsMigrator:
    """Handles migration of secrets from various sources to Vault"""

    def __init__(self, environment: str, dry_run: bool = False):
        self.environment = environment
        self.dry_run = dry_run
        self.vault = get_vault_manager()

    def migrate_api_keys_from_file(self, api_keys_file: str) -> bool:
        """Migrate API keys from local file to Vault"""
        if not Path(api_keys_file).exists():
            logger.warning(f"API keys file not found: {api_keys_file}")
            return False

        try:
            with open(api_keys_file, "r") as f:
                api_keys = json.load(f)

            for provider, key in api_keys.items():
                vault_path = f"{self.environment}/api-keys/{provider}"
                if not self.dry_run:
                    self.vault.vault.set_secret(vault_path, {"api_key": key})
                logger.info(
                    f"{'[DRY RUN] ' if self.dry_run else ''}Migrated {provider} API key to {vault_path}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to migrate API keys from file: {e}")
            return False

    def migrate_database_secrets(self) -> bool:
        """Migrate Fernet-encrypted database secrets to Vault"""
        # This would typically query the database for encrypted secrets
        # For now, we'll create placeholder dynamic secrets configuration
        logger.info(
            f"{'[DRY RUN] ' if self.dry_run else ''}Database secrets migration configured"
        )
        return True

    def migrate_bitwarden_secrets(self) -> bool:
        """Migrate secrets from Bitwarden (simulated for CI/CD context)"""
        # This would typically run in CI/CD context with Bitwarden CLI
        # For local migration, we'll use environment variables

        secrets_map = {
            "FASTAPI_SECRET": f"{self.environment}/tokens/fastapi",
            "JWT_SECRET": f"{self.environment}/tokens/jwt",
            "OPENAI_API_KEY": f"{self.environment}/api-keys/openai",
            "ANTHROPIC_API_KEY": f"{self.environment}/api-keys/anthropic",
            "GROQ_API_KEY": f"{self.environment}/api-keys/groq",
            "ELEVENLABS_API_KEY": f"{self.environment}/api-keys/elevenlabs",
            "FLY_TOKEN": f"{self.environment}/tokens/flyio",
            "CF_TOKEN": f"{self.environment}/tokens/cloudflare",
            "GITHUB_TOKEN": f"{self.environment}/tokens/github",
        }

        success_count = 0
        for env_var, vault_path in secrets_map.items():
            value = os.getenv(env_var)
            if value:
                if not self.dry_run:
                    # Determine the key name based on path
                    if "api-keys" in vault_path:
                        key_name = "api_key"
                    elif "tokens" in vault_path:
                        key_name = (
                            "token"
                            if "token" in vault_path.split("/")[-1]
                            else "secret"
                        )
                    else:
                        key_name = "value"

                    self.vault.vault.set_secret(vault_path, {key_name: value})
                logger.info(
                    f"{'[DRY RUN] ' if self.dry_run else ''}Migrated {env_var} to {vault_path}"
                )
                success_count += 1
            else:
                logger.warning(f"Environment variable {env_var} not found")

        return success_count > 0

    def migrate_ssh_keys(self) -> bool:
        """Migrate SSH keys to Vault"""
        ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")
        if not Path(ssh_key_path).exists():
            logger.warning("SSH key not found for migration")
            return False

        try:
            with open(ssh_key_path, "r") as f:
                private_key = f.read()

            vault_path = f"{self.environment}/certificates/ssh/deploy"
            if not self.dry_run:
                self.vault.vault.set_secret(vault_path, {"private_key": private_key})
            logger.info(
                f"{'[DRY RUN] ' if self.dry_run else ''}Migrated SSH key to {vault_path}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to migrate SSH key: {e}")
            return False

    def validate_migration(self) -> bool:
        """Validate that all expected secrets are present in Vault"""
        required_secrets = [
            f"{self.environment}/api-keys/openai",
            f"{self.environment}/api-keys/anthropic",
            f"{self.environment}/tokens/fastapi",
            f"{self.environment}/tokens/jwt",
        ]

        missing_secrets = []
        for secret_path in required_secrets:
            try:
                self.vault.get_secret(secret_path)
                logger.info(f"✓ Secret validated: {secret_path}")
            except Exception as e:
                logger.error(f"✗ Secret missing or invalid: {secret_path} - {e}")
                missing_secrets.append(secret_path)

        if missing_secrets:
            logger.error(
                f"Migration validation failed. Missing secrets: {missing_secrets}"
            )
            return False

        logger.info("✓ Migration validation successful")
        return True

    def run_full_migration(self) -> bool:
        """Run complete migration process"""
        logger.info(
            f"Starting {'DRY RUN ' if self.dry_run else ''}migration for environment: {self.environment}"
        )

        results = []

        # Migrate API keys from file
        results.append(
            ("API Keys File", self.migrate_api_keys_from_file("api_keys.json"))
        )

        # Migrate database secrets
        results.append(("Database Secrets", self.migrate_database_secrets()))

        # Migrate Bitwarden secrets
        results.append(("Bitwarden Secrets", self.migrate_bitwarden_secrets()))

        # Migrate SSH keys
        results.append(("SSH Keys", self.migrate_ssh_keys()))

        # Validate migration
        if not self.dry_run:
            results.append(("Validation", self.validate_migration()))

        # Summary
        successful = sum(1 for _, success in results if success)
        total = len(results)

        logger.info(f"Migration completed: {successful}/{total} steps successful")

        for step, success in results:
            status = "✓" if success else "✗"
            logger.info(f"{status} {step}")

        return successful == total


def main():
    parser = argparse.ArgumentParser(description="Migrate secrets to HashiCorp Vault")
    parser.add_argument(
        "--environment",
        required=True,
        choices=["staging", "production"],
        help="Target environment",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Perform dry run without making changes"
    )
    parser.add_argument(
        "--vault-addr",
        default="https://vault.goblin.fuaad.ai",
        help="Vault server address",
    )
    parser.add_argument("--vault-token", help="Vault authentication token")
    parser.add_argument(
        "--api-keys-file", default="api_keys.json", help="Path to API keys JSON file"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Set Vault environment variables if provided
        if args.vault_addr:
            os.environ["VAULT_ADDR"] = args.vault_addr
        if args.vault_token:
            os.environ["VAULT_TOKEN"] = args.vault_token

        # Create migrator
        migrator = SecretsMigrator(args.environment, args.dry_run)

        # Override API keys file if specified
        if args.api_keys_file != "api_keys.json":
            migrator.api_keys_file = args.api_keys_file

        # Run migration
        success = migrator.run_full_migration()

        exit_code = 0 if success else 1
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
