# Architecture

## Overview

GoblinOS Assistant is a full-stack AI gateway and orchestration platform.

## Core components

- `src/`: Next.js frontend and API proxy routes.
- `backend/`: FastAPI backend for routing, policy enforcement, and provider integration.
- `docs/`: Operational and architecture documentation.

## Runtime path

1. User interacts with frontend chat and control surfaces.
2. Frontend API routes proxy requests to backend endpoints.
3. Backend routes requests across multiple providers using policy and health signals.
4. Observability and cost instrumentation support operational tuning.
