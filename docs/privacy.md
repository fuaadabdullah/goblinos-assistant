# Privacy & Data Retention

GoblinOS Assistant stores the minimum data needed to run tasks, route
requests, and operate the service. This note documents **what is stored,
for how long**, and how to delete it.

## What is stored

| Data | Where | Contents | Retention |
|---|---|---|---|
| Task history | `tasks` | Task text, code, execution result (stdout/stderr), status, cost, tokens, duration, owning user | `DATA_RETENTION_DAYS` (default **90 days**), then purged by the daily `retention_purge` scheduler job |
| Routing log | `routing_requests` | Request id, capability, routing requirements (may include message content), selected provider, success/error, timestamp. **Not linked to a user id** | `DATA_RETENTION_DAYS` (default **90 days**), then purged |
| Search documents | `search_documents` | Document text and metadata per search collection | `DATA_RETENTION_DAYS` (default **90 days**), then purged |
| Chat messages | `chat_messages` (where present) | Anonymous (logged-out) chat messages only | **7 days** (`cleanup_expired_data` job) |
| Inference logs | `inference_logs` | Request/response logs for debugging | **30 days** (`cleanup_expired_data` job) |
| Provider metrics | `provider_metrics` | Aggregated latency/health/cost telemetry (no prompts) | **90 days** (`cleanup_expired_data` job) |
| Sessions | `user_sessions` | Login sessions | Until expiry, then removed by `cleanup_expired_data` |
| Provider credentials | `provider_credentials` | API keys **encrypted at rest** (`encrypted_key`) | Until removed by an admin |
| Account preferences | `account_prefs.json` (server volume) | Notification/summary/family-mode flags, keyed by user id | Until account deletion |

## Retention enforcement

- `backend/jobs/retention.py::purge_old_records(older_than_days)` deletes
  rows older than the retention window from `tasks`, `routing_requests`,
  and `search_documents`. The window is configured with the
  `DATA_RETENTION_DAYS` environment variable (default 90); see
  `.env.example`.
- The purge runs **daily** via the `retention_purge` APScheduler job
  (registered in `backend/scheduler.py`), with a Redis lock so only one
  replica purges at a time.
- The pre-existing `cleanup_expired_data` job (every 6 hours) handles
  inference logs, sessions, metrics, and anonymous chat messages.

## Deleting your data

- Authenticated users can erase their own data at any time with
  `DELETE /v1/account`. It deletes the user's tasks, chat history,
  provider credentials they created, sessions, stored preferences, and
  finally the user record itself, in dependency-safe order.
- `routing_requests` rows carry no user id, so they cannot be attributed
  to (or deleted for) a single user; they age out via the retention
  purge above.

## Notes & limitations

- Database backups may retain copies until the backup itself rotates;
  the purge applies to the live database.
- Retention windows are a deployment setting: self-hosted operators
  should set `DATA_RETENTION_DAYS` to match their own policy.
