#!/bin/bash
set -euo pipefail

# Startup script for GoblinOS Assistant Backend.
# Works in both local dev and containerized deployment environments.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -f "../../.env.local" ]; then
    # shellcheck disable=SC1091
    source ../../.env.local
fi

APP_ROOT="$(cd .. && pwd)"
export PYTHONPATH="${APP_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_VENV_PY="$(cd ../../../ && pwd)/.venv/bin/python3"
if [ -x "$LOCAL_VENV_PY" ]; then
    PYTHON_BIN="$LOCAL_VENV_PY"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
UVICORN_RELOAD="${UVICORN_RELOAD:-0}"
EXTRA_ARGS=""
if [ "$UVICORN_RELOAD" = "1" ]; then
    EXTRA_ARGS="--reload"
fi

echo "Starting Goblin backend with $PYTHON_BIN on ${HOST}:${PORT}"
exec "$PYTHON_BIN" -m uvicorn backend.main:app --host "$HOST" --port "$PORT" $EXTRA_ARGS
