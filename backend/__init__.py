"""
Goblin Assistant backend package.

This repo historically supported running from the `backend/` directory directly
(`uvicorn main:app`), which makes sibling modules importable as top-level
packages (e.g. `import config`).

When importing as a proper package (`import backend.main`), those absolute
imports would fail. To keep backwards compatibility, we alias `backend.config`
to the top-level module name `config` during package import.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path


def _ensure_backend_on_path() -> None:
    """Ensure backend and project root are on sys.path for legacy imports.

    This keeps flat imports like `import services` working when the backend
    is imported as a package (e.g. `python -m backend.main`).
    """

    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent
    for path in (project_root, backend_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _alias_legacy_top_level_modules() -> None:
    def _alias(alias: str, target: str, optional: bool = False) -> None:
        try:
            module = import_module(target)
            sys.modules.setdefault(alias, module)
        except Exception:
            if not optional:
                # If a core module can't import, allow package import to proceed;
                # callers will see the real error at first use.
                pass

    _alias("config", "backend.config", optional=True)
    _alias("providers", "backend.providers", optional=True)
    _alias("services", "backend.services", optional=True)
    _alias("database", "backend.database", optional=True)
    _alias("auth", "backend.auth", optional=True)
    _alias("jobs", "backend.jobs", optional=True)
    _alias("routers", "backend.routers", optional=True)
    _alias("models_base", "backend.models_base", optional=True)
    _alias("models", "backend.models", optional=True)


_ensure_backend_on_path()
_skip_legacy_aliases = str(os.getenv("BACKEND_SKIP_LEGACY_ALIASES", "")).lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if not _skip_legacy_aliases:
    _alias_legacy_top_level_modules()
