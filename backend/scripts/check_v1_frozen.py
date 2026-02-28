#!/usr/bin/env python3
"""Fail if protected v1 contract files are modified.

Usage:
  python backend/scripts/check_v1_frozen.py [<base_ref>]

Examples:
  python backend/scripts/check_v1_frozen.py origin/main
  BASE_REF=origin/main python backend/scripts/check_v1_frozen.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PROTECTED_PREFIXES = (
    "apps/goblin-assistant/backend/routers/v1/",
    "apps/goblin-assistant/backend/schemas/v1/",
)

# Documentation can evolve without being treated as contract breakage.
ALLOWED_CHANGES = {
    "apps/goblin-assistant/backend/docs/api/versioning-policy.md",
    "apps/goblin-assistant/backend/docs/api-migration-v1-v2.md",
}


def _run_git_diff(base_ref: str) -> list[str]:
    primary = ["git", "diff", "--name-only", f"{base_ref}...HEAD"]
    fallback = ["git", "diff", "--name-only", base_ref]
    for command in (primary, fallback):
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    raise RuntimeError(
        f"Unable to evaluate git diff against {base_ref}. "
        f"Tried: {' '.join(primary)} and {' '.join(fallback)}"
    )


def main() -> int:
    base_ref = sys.argv[1] if len(sys.argv) > 1 else os.getenv("BASE_REF", "origin/main")
    changed_files = _run_git_diff(base_ref)

    blocked: list[str] = []
    for path in changed_files:
        if path in ALLOWED_CHANGES:
            continue
        if any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES):
            blocked.append(path)

    if blocked:
        print("ERROR: v1 API contract is frozen. The following protected files changed:")
        for path in blocked:
            print(f" - {path}")
        print(
            "\nIf this is intentional, move the change to v2 or update governance "
            "docs and approval policy."
        )
        return 1

    print("OK: no protected v1 contract files changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
