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

### Internal proxy key (Next.js → FastAPI)

The Vercel frontend proxies guest-chat traffic to the FastAPI backend through
`src/pages/api/*`. Every proxied request carries the header
`X-Internal-API-Key`, which the backend validates in
`require_internal_proxy_key` (`backend/auth/dependencies.py`).

- Backend accepts the first non-empty of `INTERNAL_PROXY_API_KEY`,
  `BACKEND_API_KEY`, `INTERNAL_API_SECRET` (in that order). The frontend reads
  the same three names in the same order.
- **The same secret value must be set on both sides**: Render dashboard
  (backend service) **and** Vercel dashboard (frontend project, server-side
  only — never expose it as `NEXT_PUBLIC_*`).
- Fail-closed: the backend returns `401` on the proxied endpoints when the key
  is missing or wrong, and **refuses to boot in production** when none of the
  three env vars is set.

Generate a strong secret with:

```bash
python3 scripts/generate_internal_proxy_key.py
```

(`openssl rand -base64 32` works too.) Paste the output into both dashboards.

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
