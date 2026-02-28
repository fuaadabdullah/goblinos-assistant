# GoblinOS Assistant

Canonical repository for GoblinOS Assistant.

GoblinOS Assistant is a multi-provider, privacy-first AI gateway and orchestration platform with a full-stack implementation (Next.js + FastAPI).

## Core capabilities

- Multi-provider model routing and failover
- Cost and latency visibility
- Chat and orchestration interfaces
- Policy-aware gateway behavior

## Stack

- Frontend: Next.js, TypeScript, React, Tailwind
- Backend: FastAPI, Python
- Data and infra: PostgreSQL, Redis, Docker, Cloud deployment workflows

## Quickstart

```bash
pnpm install
pnpm dev
```

Backend dependencies and startup:

```bash
python3 -m venv venv
./venv/bin/pip install -r backend/requirements.txt
./start.sh
```

## Docs

- [Architecture](docs/architecture.md)
- [Setup](docs/setup.md)
- [Impact](docs/impact.md)
