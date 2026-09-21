#!/usr/bin/env python3
"""Generate a strong internal proxy key for the Next.js -> FastAPI proxy.

The same value must be set as INTERNAL_PROXY_API_KEY (or BACKEND_API_KEY /
INTERNAL_API_SECRET) on the Render backend service AND the Vercel frontend
project. The backend refuses to boot in production without one of these set.

Usage: python3 scripts/generate_internal_proxy_key.py
"""

from __future__ import annotations

import secrets


def main() -> None:
    print(secrets.token_urlsafe(32))


if __name__ == "__main__":
    main()
