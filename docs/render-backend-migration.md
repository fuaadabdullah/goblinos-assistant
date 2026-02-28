# Render Backend Migration Runbook

This runbook migrates `apps/goblin-assistant` backend compute from Fly.io to Render while keeping Fly available for rollback.

## Scope
- Backend compute moves to Render.
- Existing external services remain unchanged: PostgreSQL, Redis, model providers.
- Frontend remains on Vercel.

## Prerequisites
- Render account with access to the target team/project.
- Repo connected to Render.
- All required production secrets available.
- Existing Fly deployment still healthy (rollback safety).

## 1. Create Render Service From Blueprint
1. In Render, create a Blueprint from this repository.
2. Confirm service `goblin-backend` from [`render.yaml`](../render.yaml).
3. Confirm region is `virginia` and plan is `starter`.

## 2. Populate Required Secrets In Render
Set these required secrets in Render service settings:
- `DATABASE_URL`
- `REDIS_URL`
- `JWT_SECRET_KEY`
- `ROUTING_ENCRYPTION_KEY`
- `SETTINGS_ENCRYPTION_KEY`
- `INTERNAL_PROXY_API_KEY`
- `CORS_ORIGINS`

Recommended `CORS_ORIGINS` value:
- `https://goblin-assistant.vercel.app,https://goblin-assistant-*.vercel.app,http://localhost:3000,http://localhost:5173`

Set provider-dependent secrets only if used:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `TAVILY_API_KEY`
- `OLLAMA_GCP_URL`, `LLAMACPP_GCP_URL`, `LOCAL_LLM_API_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`, `SENTRY_DSN`

## 3. Deploy And Validate Render Backend
1. Trigger a deploy from Render dashboard or via deploy hook.
2. Verify backend health:
```bash
curl -fsS https://goblin-backend.onrender.com/health
```
Expected payload:
```json
{"status":"healthy"}
```
If you get HTML or `{"status":"ok"}`, the domain is pointing to the wrong service.

3. Verify proxy-auth endpoints with internal key:
```bash
curl -fsS -H "X-Internal-API-Key: $INTERNAL_PROXY_API_KEY" \
  https://goblin-backend.onrender.com/v1/providers/models
```

## 4. Gradual Cutover To Render
1. Update Vercel project env vars:
- `NEXT_PUBLIC_API_BASE_URL=https://goblin-backend.onrender.com`
- `NEXT_PUBLIC_BACKEND_URL=https://goblin-backend.onrender.com`
- `GOBLIN_BACKEND_URL=https://goblin-backend.onrender.com`
2. Redeploy frontend.
3. Validate end-to-end behavior from production UI:
- chat/generate flow
- models listing
- auth flows

## 5. Post-Cutover Monitoring
- Monitor Render service health and logs.
- Watch application metrics/errors (Sentry/Datadog as configured).
- Keep Fly running during stabilization window.

## Rollback (Fly)
If issues occur after cutover:
1. Revert Vercel env vars back to Fly backend URL:
- `https://goblin-backend.fly.dev`
2. Redeploy frontend.
3. Confirm health:
```bash
curl -fsS https://goblin-backend.fly.dev/health
```
4. If needed, run rollback deployment workflow:
- GitHub Actions: `Fly Rollback Deploy (Deprecated)` workflow (manual trigger)

## Optional Decommission (After Stabilization)
1. Confirm sustained stability on Render.
2. Disable Fly auto deployments (already manual-only in workflow).
3. Shut down Fly backend when rollback window is no longer required.
