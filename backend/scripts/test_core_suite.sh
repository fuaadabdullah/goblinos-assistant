#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m pytest --collect-only tests -m "not optional and not integration and not e2e and not slow" -q
python3 -m pytest tests -m "not optional and not integration and not e2e and not slow" -q

