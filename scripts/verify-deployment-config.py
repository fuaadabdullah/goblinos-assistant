#!/usr/bin/env python3
"""
Comprehensive Vercel Deployment Configuration Checker
Verifies all configuration before deployment to prevent errors.
"""

import os
import sys
import json
import subprocess
from pathlib import Path


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"


def check_mark(passed):
    return f"{Colors.GREEN}✅{Colors.END}" if passed else f"{Colors.RED}❌{Colors.END}"


def warning_mark():
    return f"{Colors.YELLOW}⚠️ {Colors.END}"


def print_section(title):
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.END}")


def check_vercel_cli():
    """Check if Vercel CLI is installed and authenticated"""
    print_section("1. Vercel CLI Check")

    try:
        result = subprocess.run(
            ["vercel", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"{check_mark(True)} Vercel CLI installed: {version}")
            return True
        else:
            print(f"{check_mark(False)} Vercel CLI not working")
            return False
    except Exception as e:
        print(f"{check_mark(False)} Vercel CLI not found: {e}")
        print("   Install with: npm i -g vercel")
        return False


def check_vercel_auth():
    """Check if authenticated with Vercel"""
    try:
        config_path = os.path.expanduser("~/.config/vercel/auth.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                if config.get("token"):
                    print(f"{check_mark(True)} Vercel authenticated")
                    return True

        print(f"{check_mark(False)} Not authenticated with Vercel")
        print("   Run: vercel login")
        return False
    except Exception as e:
        print(f"{check_mark(False)} Could not verify auth: {e}")
        return False


def check_project_linked():
    """Check if project is linked to Vercel"""
    print_section("2. Project Configuration")

    project_file = ".vercel/project.json"
    if os.path.exists(project_file):
        try:
            with open(project_file) as f:
                data = json.load(f)
                project_id = data.get("projectId")
                org_id = data.get("orgId")

                print(f"{check_mark(True)} Project linked")
                print(f"   Project ID: {project_id}")
                if org_id:
                    print(f"   Team ID: {org_id}")
                return True
        except Exception as e:
            print(f"{check_mark(False)} Invalid project config: {e}")
            return False
    else:
        print(f"{check_mark(False)} Project not linked")
        print("   Run: vercel link")
        return False


def check_vercel_json():
    """Check vercel.json configuration"""
    if os.path.exists("vercel.json"):
        try:
            with open("vercel.json") as f:
                config = json.load(f)

                print(f"{check_mark(True)} vercel.json found")

                # Check important fields
                if "buildCommand" in config:
                    print(f"   Build: {config['buildCommand']}")

                if "installCommand" in config:
                    print(f"   Install: {config['installCommand']}")

                if "rewrites" in config:
                    print(f"   {len(config['rewrites'])} API rewrites configured")

                if "env" in config:
                    print(f"   {len(config['env'])} environment variables defined")

                return True
        except Exception as e:
            print(f"{check_mark(False)} Invalid vercel.json: {e}")
            return False
    else:
        print(f"{warning_mark()} vercel.json not found (optional)")
        return True


def check_next_config():
    """Check Next.js configuration"""
    print_section("3. Next.js Configuration")

    configs = ["next.config.js", "next.config.mjs"]
    config_found = False

    for config_file in configs:
        if os.path.exists(config_file):
            config_found = True
            print(f"{check_mark(True)} {config_file} found")

            with open(config_file) as f:
                content = f.read()

                # Check for important configurations
                checks = {
                    "output": "standalone" in content or "output:" in content,
                    "rewrites": "rewrites" in content or "async rewrites" in content,
                    "env vars": "NEXT_PUBLIC_API_URL" in content,
                }

                for check_name, passed in checks.items():
                    print(f"   {check_mark(passed)} {check_name}")

            break

    if not config_found:
        print(f"{check_mark(False)} No Next.js config found")
        return False

    return True


def check_package_json():
    """Check package.json for required scripts"""
    if os.path.exists("package.json"):
        try:
            with open("package.json") as f:
                package = json.load(f)

                print(f"{check_mark(True)} package.json found")

                scripts = package.get("scripts", {})
                required_scripts = ["build", "dev"]

                for script in required_scripts:
                    if script in scripts:
                        print(f"   {check_mark(True)} {script}: {scripts[script]}")
                    else:
                        print(f"   {check_mark(False)} Missing '{script}' script")

                return True
        except Exception as e:
            print(f"{check_mark(False)} Invalid package.json: {e}")
            return False
    else:
        print(f"{check_mark(False)} package.json not found")
        return False


def check_env_files():
    """Check environment files"""
    print_section("4. Environment Configuration")

    env_files = {
        ".env.example": "Example file",
        ".env": "Development file (should NOT be committed)",
        ".env.local": "Local overrides (should NOT be committed)",
    }

    for env_file, description in env_files.items():
        if os.path.exists(env_file):
            print(f"{check_mark(True)} {env_file} - {description}")
        else:
            if env_file == ".env.example":
                print(f"{warning_mark()} {env_file} missing (recommended)")
            else:
                print(f"   ℹ️  {env_file} not present")


def check_required_env_vars():
    """Check if required env vars are in config"""
    required_vars = [
        "NEXT_PUBLIC_API_URL",
        "NEXT_PUBLIC_FASTAPI_URL",
        "NEXT_PUBLIC_DD_APPLICATION_ID",
        "NEXT_PUBLIC_DD_ENV",
    ]

    print(f"\n   Required variables in vercel.json:")

    if os.path.exists("vercel.json"):
        try:
            with open("vercel.json") as f:
                config = json.load(f)
                env_config = config.get("env", {})

                for var in required_vars:
                    if var in env_config:
                        print(f"   {check_mark(True)} {var}")
                    else:
                        print(f"   {check_mark(False)} {var} missing")
        except:
            pass


def check_gitignore():
    """Check .gitignore for important entries"""
    print_section("5. Git Configuration")

    if os.path.exists(".gitignore"):
        with open(".gitignore") as f:
            content = f.read()

        important_entries = [
            (".env", "Environment files"),
            (".vercel", "Vercel config"),
            ("node_modules", "Dependencies"),
        ]

        print(f"{check_mark(True)} .gitignore found")

        for entry, description in important_entries:
            if entry in content:
                print(f"   {check_mark(True)} {entry} - {description}")
            else:
                print(f"   {warning_mark()} {entry} not in .gitignore")
    else:
        print(f"{check_mark(False)} .gitignore not found")


def check_vercelignore():
    """Check .vercelignore"""
    if os.path.exists(".vercelignore"):
        with open(".vercelignore") as f:
            lines = [
                l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")
            ]

        print(f"{check_mark(True)} .vercelignore found ({len(lines)} exclusions)")

        important = ["*.md", "backend/", "*.py", "tests/", "docs/"]
        found = sum(
            1 for pattern in important if any(pattern in line for line in lines)
        )
        print(f"   Excludes {found}/{len(important)} recommended patterns")
    else:
        print(f"{warning_mark()} .vercelignore not found")


def check_api_proxy():
    """Check if API proxy is configured"""
    print_section("6. API Proxy Configuration")

    backend_url = "https://goblin-backend.onrender.com"
    proxy_configured = False

    # Check next.config
    for config_file in ["next.config.js", "next.config.mjs"]:
        if os.path.exists(config_file):
            with open(config_file) as f:
                content = f.read()
                if "rewrites" in content and backend_url in content:
                    print(f"{check_mark(True)} Proxy configured in {config_file}")
                    proxy_configured = True

    # Check vercel.json
    if os.path.exists("vercel.json"):
        with open("vercel.json") as f:
            config = json.load(f)
            if "rewrites" in config:
                rewrites = config["rewrites"]
                api_routes = [r for r in rewrites if "/api" in r.get("source", "")]
                if api_routes:
                    print(
                        f"{check_mark(True)} {len(rewrites)} API routes in vercel.json"
                    )
                    proxy_configured = True

    if not proxy_configured:
        print(f"{check_mark(False)} No API proxy configured")
        print(f"   Backend won't be accessible from frontend")

    return proxy_configured


def check_dependencies():
    """Check if dependencies are installed"""
    print_section("7. Dependencies")

    if os.path.exists("node_modules"):
        print(f"{check_mark(True)} node_modules exists")

        # Count packages
        try:
            package_count = len(
                [
                    d
                    for d in os.listdir("node_modules")
                    if os.path.isdir(os.path.join("node_modules", d))
                ]
            )
            print(f"   ~{package_count} packages installed")
        except:
            pass
    else:
        print(f"{warning_mark()} node_modules not found")
        print("   Run: pnpm install")


def check_build():
    """Check if the project builds"""
    print_section("8. Build Check")

    if os.path.exists(".next"):
        print(f"{check_mark(True)} .next directory exists (previous build)")
    else:
        print(f"   ℹ️  No previous build found")

    print(f"   {warning_mark()} Run 'pnpm build' to verify build works")


def main():
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BLUE}🔍 Vercel Deployment Configuration Checker{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.END}")

    app_root = Path(__file__).resolve().parents[1]
    os.chdir(app_root)

    checks = [
        check_vercel_cli(),
        check_vercel_auth(),
        check_project_linked(),
        check_vercel_json(),
        check_next_config(),
        check_package_json(),
    ]

    check_env_files()
    check_required_env_vars()
    check_gitignore()
    check_vercelignore()

    checks.append(check_api_proxy())

    check_dependencies()
    check_build()

    # Summary
    print_section("Summary")
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"\n{Colors.GREEN}✅ All checks passed! Ready to deploy.{Colors.END}")
        print(f"\nNext steps:")
        print(f"   1. Run: python3 setup-vercel-env.py")
        print(f"   2. Run: vercel deploy --prod")
        return 0
    else:
        print(f"\n{Colors.YELLOW}⚠️  {passed}/{total} checks passed{Colors.END}")
        print(f"\nFix the issues above before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
