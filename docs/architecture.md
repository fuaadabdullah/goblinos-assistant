# Architecture

## Overview

GoblinOS Assistant is a full-stack AI gateway and orchestration platform with a web client, API proxy layer, and backend provider routing service.

## Core components

- `src/`: Next.js frontend, route handlers, and UI surfaces (chat/settings).
- `backend/`: FastAPI services for routing, policy controls, and provider integrations.
- `config/`: deployment/runtime config files (`render.yaml`, `fly.toml`, etc.).
- `docs/`: architecture, setup, migration, and impact notes.

## Request flow

1. User sends a message from frontend chat interface.
2. Next.js route layer forwards request to backend API.
3. Backend evaluates provider strategy and health signals.
4. Selected provider executes inference request.
5. Backend returns normalized response + telemetry signals.
6. Frontend renders output and status indicators.

## Deployment topology

- Frontend deployed on Vercel.
- Backend deployed on Render/Fly-compatible runtime.
- Redis + PostgreSQL provide cache/session/data dependencies.
- External LLM providers serve model inference workloads.

## Reliability controls

- Provider failover and routing strategy controls.
- Environment-separated secrets for backend and frontend.
- Operational toggles for feature and licensing behavior.
