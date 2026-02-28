"""Compatibility exports for generate request schemas.

Supports both runtime layouts used in this repo:
- backend root on PYTHONPATH (import from `schemas`)
- repo root on PYTHONPATH (import from `backend.schemas`)
"""

try:  # backend root import layout
    from schemas.v1.generate import GenerateMessage, GenerateRequest
except ImportError:  # repo root import layout
    from backend.schemas.v1.generate import GenerateMessage, GenerateRequest

__all__ = ["GenerateMessage", "GenerateRequest"]
