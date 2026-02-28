"""Compatibility shim: allow imports like `backend.services...` when tests are
executed with the current working directory set to the `backend/` folder.

When pytest is run from inside `backend/`, importing `backend.` fails because
the importer looks for a *backend* package inside the current directory.
This module makes the `backend` module act like a package by exposing a
`__path__` pointing at the package directory so submodules (e.g. `services`)
can be imported as `backend.services.*`.

This file is a harmless local compatibility shim and does not change
normal import behavior when the project root is on PYTHONPATH.
"""

import os

# Make this module act like a package by exposing __path__ so
# `import backend.services...` works when cwd == <project>/apps/goblin-assistant-root/backend
__path__ = [os.path.dirname(__file__)]

# Re-export package metadata if present
try:
    # If the real package __init__ is importable, prefer its attributes
    from . import __all__ as __all__  # type: ignore
except Exception:
    __all__ = ["services", "config", "database"]
