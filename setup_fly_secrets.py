#!/usr/bin/env python3
"""
Fly.io Secrets Manager for Goblin Assistant

DEPRECATED: Fly.io deployment is rollback-only.
Primary backend deployment target is Render.

Manages required secrets for Fly.io deployment:
- OLLAMA_GCP_URL: GCP-hosted Ollama instance endpoint
- LLAMACPP_GCP_URL: GCP-hosted Llama.cpp server endpoint
- SUPABASE_SERVICE_ROLE_KEY: Supabase database access
- RAG_API_KEY: RAG pipeline key

Usage:
    # List current secrets (not values, just names)
    python setup_fly_secrets.py list

    # Set all required secrets interactively
    python setup_fly_secrets.py set

    # Set a specific secret
    python setup_fly_secrets.py set --key OLLAMA_GCP_URL --value "https://34.28.123.45:11434"

    # Validate secrets exist
    python setup_fly_secrets.py validate

    # Get setup status
    python setup_fly_secrets.py status
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional, List


class FlySecretsManager:
    """Manage Fly.io secrets for rollback-only Goblin Assistant deployments."""

    REQUIRED_SECRETS = [
        "OLLAMA_GCP_URL",  # GCP Ollama endpoint (e.g., https://34.28.123.45:11434)
        "LLAMACPP_GCP_URL",  # GCP Llama.cpp endpoint (e.g., http://34.123.36.255:8080)
        "SUPABASE_SERVICE_ROLE_KEY",  # Database access token
        "RAG_API_KEY",  # RAG pipeline API key
    ]

    OPTIONAL_SECRETS = [
        "GCP_OLLAMA_URL",  # Fallback Ollama URL
        "GCP_LLAMACPP_URL",  # Fallback Llama.cpp URL
        "TOGETHER_API_KEY",  # Together AI API key
        "OPENROUTER_API_KEY",  # OpenRouter API key
        "DEEPINFRA_API_KEY",  # DeepInfra API key
        "GROQ_API_KEY",  # Groq API key
        "AZURE_API_KEY",  # Azure OpenAI API key
        "AZURE_OPENAI_ENDPOINT",  # Azure OpenAI endpoint URL
        "OPENAI_API_KEY",  # OpenAI API key fallback
        "ALIYUN_MODEL_SERVER_URL",  # Aliyun model server endpoint
        "ALIYUN_MODEL_SERVER_KEY",  # Aliyun model server API key
    ]

    def __init__(self, app_name: str = "goblin-backend"):
        """Initialize secrets manager"""
        self.app_name = app_name
        self._check_fly_cli()

    def _check_fly_cli(self) -> bool:
        """Check if Fly.io CLI is installed and authenticated"""
        try:
            result = subprocess.run(
                ["fly", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                print("❌ Fly.io CLI not found or not authenticated")
                print("   Install: curl -L https://fly.io/install.sh | sh")
                print("   Authenticate: fly auth login")
                return False
            print(f"✅ Fly.io CLI: {result.stdout.strip()}")
            return True
        except Exception as e:
            print(f"❌ Error checking Fly CLI: {e}")
            return False

    def list_secrets(self) -> Dict[str, bool]:
        """List existing secrets (names only, not values)"""
        try:
            result = subprocess.run(
                ["fly", "secrets", "list", "-a", self.app_name, "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                # Try non-JSON output if JSON fails
                result = subprocess.run(
                    ["fly", "secrets", "list", "-a", self.app_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    secrets = {}
                    for line in result.stdout.strip().split("\n"):
                        if line:
                            secrets[line.split()[0]] = True
                    return secrets
                return {}

            # Parse JSON output
            try:
                secrets = json.loads(result.stdout)
                return {s["name"]: True for s in secrets if isinstance(secrets, list)}
            except:
                # Fallback to text parsing
                secrets = {}
                for line in result.stdout.strip().split("\n"):
                    if line and "=" in line:
                        key = line.split("=")[0].strip()
                        secrets[key] = True
                return secrets

        except subprocess.TimeoutExpired:
            print("❌ Timeout listing secrets")
            return {}
        except Exception as e:
            print(f"❌ Error listing secrets: {e}")
            return {}

    def set_secret(self, key: str, value: str) -> bool:
        """Set a single secret"""
        try:
            result = subprocess.run(
                ["fly", "secrets", "set", f"{key}={value}", "-a", self.app_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                print(f"✅ Secret set: {key}")
                return True
            else:
                print(f"❌ Failed to set {key}: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error setting secret {key}: {e}")
            return False

    def validate_secrets(self) -> Dict[str, bool]:
        """Check which required secrets are set"""
        existing = self.list_secrets()
        status = {}

        print("\n📋 Checking required secrets:")
        for secret in self.REQUIRED_SECRETS:
            is_set = secret in existing
            status[secret] = is_set
            icon = "✅" if is_set else "❌"
            print(f"  {icon} {secret}")

        print("\n📋 Checking optional secrets:")
        for secret in self.OPTIONAL_SECRETS:
            is_set = secret in existing
            icon = "✅" if is_set else "⊘"
            print(f"  {icon} {secret}")

        return status

    def setup_interactive(self) -> None:
        """Interactive setup wizard"""
        print("\n🚀 Fly.io Secrets Setup Wizard")
        print("=" * 50)

        existing = self.list_secrets()

        # Required secrets
        print("\n📝 Required Secrets:")
        print("-" * 50)

        secrets_to_set = {}

        for secret in self.REQUIRED_SECRETS:
            if secret in existing:
                print(f"✅ {secret} (already set)")
                continue

            print(f"\n❓ {secret}")

            if secret == "OLLAMA_GCP_URL":
                example = "https://34.28.123.45:11434"
                print(f"   (GCP Ollama instance URL, e.g., {example})")
            elif secret == "LLAMACPP_GCP_URL":
                example = "http://34.123.36.255:8080"
                print(f"   (GCP Llama.cpp server URL, e.g., {example})")
            elif secret == "SUPABASE_SERVICE_ROLE_KEY":
                print("   (From Supabase dashboard > Project Settings > API)")
            elif secret == "RAG_API_KEY":
                print("   (Any secure key, used internally for RAG)")

            value = input(f"   Enter {secret}: ").strip()
            if value:
                secrets_to_set[secret] = value
            else:
                print(f"   ⊘ Skipped {secret}")

        # Optional secrets
        if input("\n📝 Set optional secrets? (y/n): ").lower() == "y":
            print("\n(Press Enter to skip any secret)")
            for secret in self.OPTIONAL_SECRETS:
                if secret in existing:
                    print(f"✅ {secret} (already set)")
                    continue

                value = input(f"   Enter {secret} (optional): ").strip()
                if value:
                    secrets_to_set[secret] = value

        # Confirm and set
        if secrets_to_set:
            print("\n📋 Summary of secrets to set:")
            for key, value in secrets_to_set.items():
                # Show partial value for security
                display_val = value[:10] + "..." if len(value) > 10 else value
                print(f"  - {key}: {display_val}")

            if input("\n✅ Confirm setup? (y/n): ").lower() == "y":
                print("\n🔧 Setting secrets...")
                for key, value in secrets_to_set.items():
                    self.set_secret(key, value)

                print("\n✅ All secrets set!")
                print("\nNext steps (rollback-only path):")
                print("1. Deploy to Fly.io rollback target: fly deploy")
                print("2. Check logs: fly logs")
                print(
                    "3. Verify providers: curl https://goblin-backend.fly.dev/routing/providers/details"
                )
            else:
                print("❌ Setup cancelled")
        else:
            print("⊘ No secrets to set")

    def get_status(self) -> None:
        """Show comprehensive setup status"""
        print("\n📊 Goblin Assistant on Fly.io - Setup Status")
        print("=" * 50)

        # Check Fly CLI
        print("\n🔍 Environment:")
        result = subprocess.run(
            ["fly", "whoami"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  ✅ Fly CLI: Authenticated as {result.stdout.strip()}")
        else:
            print("  ❌ Fly CLI: Not authenticated")

        # Check app
        result = subprocess.run(
            ["fly", "info", "-a", self.app_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print(f"  ✅ Fly App: {self.app_name} exists")
        else:
            print(f"  ❌ Fly App: {self.app_name} not found")

        # Check secrets
        status = self.validate_secrets()

        required_set = sum(1 for s in self.REQUIRED_SECRETS if status.get(s, False))
        print(f"\n  Secrets: {required_set}/{len(self.REQUIRED_SECRETS)} required")

        # Recommendations
        print("\n💡 Recommendations:")
        missing = [s for s in self.REQUIRED_SECRETS if not status.get(s, False)]
        if missing:
            print(f"  ⚠️  Missing {len(missing)} required secret(s):")
            for secret in missing:
                print(f"     - {secret}")
            print("\n  Run: python setup_fly_secrets.py set")
        else:
            print("  ✅ All required secrets are configured!")


def main():
    """Main entry point"""
    print("⚠️  DEPRECATED: Fly.io is rollback-only. Primary backend target is Render.")

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    manager = FlySecretsManager()

    if command == "list":
        print("\n📝 Current secrets:")
        secrets = manager.list_secrets()
        if secrets:
            for key in sorted(secrets.keys()):
                print(f"  - {key}")
        else:
            print("  (no secrets set)")

    elif command == "set":
        if len(sys.argv) > 3 and sys.argv[2] == "--key":
            # Set specific secret via CLI args
            key = sys.argv[3]
            value = (
                sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--value" else None
            )
            if not value:
                value = input(f"Enter value for {key}: ").strip()
            manager.set_secret(key, value)
        else:
            # Interactive setup
            manager.setup_interactive()

    elif command == "validate":
        manager.validate_secrets()

    elif command == "status":
        manager.get_status()

    else:
        print(f"❌ Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
