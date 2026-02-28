#!/bin/bash
set -e

echo "🚀 Starting Goblin Assistant Backend..."

# TinyLlama is intentionally not installed on Fly.
# Local model workloads should run on dedicated model servers (e.g. GCP Ollama/Llama.cpp).
# Keep local ML disabled by default in API deployments.
export DISABLE_LOCAL_ML="${DISABLE_LOCAL_ML:-true}"
export ENABLE_ENHANCED_RAG="${ENABLE_ENHANCED_RAG:-false}"

# Set Python path
export PYTHONPATH=/app

# Start uvicorn with production settings
echo "🎯 Starting Uvicorn server on port ${PORT:-8001}..."
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"
UVICORN_LOG_LEVEL="$(printf '%s' "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"
exec uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8001} \
    --workers "${UVICORN_WORKERS}" \
    --log-level "${UVICORN_LOG_LEVEL}" \
    --no-access-log
