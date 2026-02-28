"""Lightweight services package initializer.

This package intentionally avoids eager imports to prevent import-time cycles.
Callers should import concrete modules directly when possible.
"""

from importlib import import_module


def __getattr__(name: str):
    """Lazily resolve `backend.services.<module>` attributes."""
    try:
        module = import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    globals()[name] = module
    return module


def __dir__():
    return sorted(set(globals().keys()))

