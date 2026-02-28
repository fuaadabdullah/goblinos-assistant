# GoblinOS Assistant

Canonical repository for GoblinOS Assistant.

GoblinOS Assistant is a multi-provider, privacy-first AI gateway and orchestration platform with a full-stack implementation (Next.js + FastAPI).

## Value proposition

GoblinOS Assistant centralizes model access, routing, and reliability controls so teams can ship AI features faster without vendor lock-in.

## Core capabilities

- Multi-provider model routing and failover
- Cost and latency visibility
- Chat and orchestration interfaces
- Policy-aware gateway behavior

## Stack

- Frontend: Next.js, TypeScript, React, Tailwind
- Backend: FastAPI, Python
- Data and infra: PostgreSQL, Redis, Docker, Cloud deployment workflows

## Architecture

System design, request flow, and component boundaries are documented in [docs/architecture.md](docs/architecture.md).

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

## Deployment

Deployment environment assumptions and runbook details are in [docs/setup.md](docs/setup.md).

## Impact

Business and engineering outcomes are documented in [docs/impact.md](docs/impact.md).

## Docs

- [Architecture](docs/architecture.md)
- [Setup](docs/setup.md)
- [Impact](docs/impact.md)

## Contact

- Email: fuaadabdullah@gmail.com
- LinkedIn: https://www.linkedin.com/in/fuaadabdullah
