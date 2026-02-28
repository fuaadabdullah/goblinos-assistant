# Setup

## Prerequisites

- Node.js 18+
- pnpm
- Python 3.11+
- PostgreSQL
- Redis

## Environment configuration

Copy `.env.example` to `.env.local` and set values.

Critical variables:

- `DATABASE_URL`
- `REDIS_URL`
- `JWT_SECRET_KEY`
- `INTERNAL_PROXY_API_KEY`
- provider credentials (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)

## Frontend local run

```bash
pnpm install
pnpm dev
```

## Backend local run

```bash
python3 -m venv venv
./venv/bin/pip install -r backend/requirements.txt
./start.sh
```

## Validation

```bash
pnpm lint
pnpm test
```

## Deploy

- Frontend: deploy on Vercel with server-side secrets configured.
- Backend: deploy via Render/Fly config and set required environment variables.
- Confirm CORS and proxy keys are aligned between frontend and backend.

## Troubleshooting

- `401/403` from backend: check `INTERNAL_PROXY_API_KEY` parity.
- provider failures: verify provider API keys and model availability.
- routing anomalies: check strategy env vars and provider health logs.
