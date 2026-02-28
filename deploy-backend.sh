#!/bin/bash

set -euo pipefail

# Goblin Assistant Backend Production Deployment Script
# Primary target: Render
# Fly.io support is retained for rollback only.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_BASE="${TMPDIR:-/tmp}"
if [[ ! -d "$TMP_BASE" ]]; then
  TMP_BASE="/tmp"
fi

APP_NAME="${FLY_APP_NAME:-goblin-backend}"
FLY_HEALTH_URL="${FLY_HEALTHCHECK_URL:-https://goblin-backend.fly.dev/health}"
RENDER_BACKEND_URL="${RENDER_BACKEND_URL:-https://goblin-backend.onrender.com}"
RENDER_HEALTH_URL="${RENDER_HEALTHCHECK_URL:-${RENDER_BACKEND_URL%/}/health}"
RENDER_DEPLOY_HOOK_URL="${RENDER_DEPLOY_HOOK_URL:-}"

STAGING_DIR=""
LAST_DEPLOY_LOG=""

print_status() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
  echo -e "${BLUE}[STEP]${NC} $1"
}

cleanup_staging_context() {
  if [[ -z "${STAGING_DIR:-}" || ! -d "$STAGING_DIR" ]]; then
    return
  fi

  if [[ "${KEEP_FLY_CONTEXT:-0}" == "1" ]]; then
    print_warning "Keeping staged context at: $STAGING_DIR (KEEP_FLY_CONTEXT=1)"
    return
  fi

  rm -rf "$STAGING_DIR"
}

is_transport_failure_log() {
  local log_file="$1"
  grep -Eqi \
    "unable to upgrade to h2c|received 500|remote builder|transport|stream error|connection reset by peer|tls handshake timeout|i/o timeout|context canceled" \
    "$log_file"
}

run_fly_deploy_attempt() {
  local attempt_label="$1"
  shift

  LAST_DEPLOY_LOG="$(mktemp "$TMP_BASE/fly-deploy-${attempt_label}.XXXXXX.log")"
  print_status "Deploy attempt ${attempt_label}: fly deploy $*"

  set +e
  fly deploy "$STAGING_DIR" --app "$APP_NAME" --remote-only --yes "$@" 2>&1 | tee "$LAST_DEPLOY_LOG"
  local exit_code=${PIPESTATUS[0]}
  set -e

  return "$exit_code"
}

verify_fly_deployment() {
  print_step "Verifying Fly deployment status..."
  fly status --app "$APP_NAME"

  print_step "Running Fly health probe..."
  if curl -fsS --max-time 30 "$FLY_HEALTH_URL" >/dev/null; then
    print_status "Health check passed: $FLY_HEALTH_URL"
    return
  fi

  print_error "Health check failed: $FLY_HEALTH_URL"
  print_warning "Recent logs:"
  fly logs --app "$APP_NAME" --lines 80 || true
  exit 1
}

verify_render_deployment() {
  local retries="${RENDER_HEALTH_RETRIES:-12}"
  local interval_s="${RENDER_HEALTH_INTERVAL_SECONDS:-10}"
  local response=""

  print_step "Running Render health probe at: $RENDER_HEALTH_URL"

  for attempt in $(seq 1 "$retries"); do
    response="$(curl -fsS --max-time 30 "$RENDER_HEALTH_URL" 2>/dev/null || true)"
    if [[ -n "$response" ]] && grep -Eiq '"status"[[:space:]]*:[[:space:]]*"healthy"' <<<"$response"; then
      print_status "Health check passed: $RENDER_HEALTH_URL"
      return
    fi

    if [[ -n "$response" ]]; then
      print_warning "Render health response did not match expected backend payload: $response"
    fi
    print_warning "Render health probe failed (attempt ${attempt}/${retries}); retrying in ${interval_s}s"
    sleep "$interval_s"
  done

  print_error "Render health check failed after ${retries} attempts: $RENDER_HEALTH_URL"
  print_warning 'Expected payload must include: {"status":"healthy"}'
  print_status "Check Render dashboard logs for service: goblin-backend"
  exit 1
}

manual_fly_instructions() {
  echo ""
  print_status "Manual Fly.io Rollback Instructions:"
  echo "1. Go to https://fly.io"
  echo "2. Install Fly CLI: curl -L https://fly.io/install.sh | sh"
  echo "3. Login: fly auth login"
  echo "4. Create app: fly launch --name $APP_NAME --region iad --no-deploy"
  echo "5. Create volume: fly volumes create data --region iad --size 10"
  echo "6. Set secrets: fly secrets set <KEY>=<VALUE> for each env var"
  echo "7. Deploy: fly deploy"
  echo ""
}

prepare_staged_context() {
  local prepare_script="$SCRIPT_DIR/scripts/fly_prepare_context.sh"
  if [[ ! -x "$prepare_script" ]]; then
    print_error "Missing executable context script: $prepare_script"
    exit 1
  fi

  STAGING_DIR="$("$prepare_script")"
  print_status "Using staged build context: $STAGING_DIR"
}

deploy_to_fly() {
  print_warning "Fly.io deployment path is DEPRECATED and retained for rollback only."
  print_step "Preparing Fly.io rollback deployment..."

  if ! command -v fly >/dev/null 2>&1; then
    print_warning "Fly CLI not found. Please install it:"
    print_status "curl -L https://fly.io/install.sh | sh"
    manual_fly_instructions
    return
  fi

  if [[ ! -f "$SCRIPT_DIR/fly.toml" ]]; then
    print_warning "fly.toml not found in $SCRIPT_DIR."
    print_status "Create it first with: fly launch --no-deploy"
    manual_fly_instructions
    return
  fi

  export COPYFILE_DISABLE=1
  export COPY_EXTENDED_ATTRIBUTES_DISABLE=1

  prepare_staged_context

  if run_fly_deploy_attempt "1" --depot=true; then
    print_status "Fly deploy succeeded on attempt 1."
    verify_fly_deployment
    return
  fi

  if ! is_transport_failure_log "$LAST_DEPLOY_LOG"; then
    print_error "Deploy failed with non-transport error on attempt 1."
    print_warning "See deploy log: $LAST_DEPLOY_LOG"
    exit 1
  fi

  print_warning "Transport-level builder failure detected. Retrying in 10s with builder recreation."
  sleep 10

  if run_fly_deploy_attempt "2" --depot=false --recreate-builder; then
    print_status "Fly deploy succeeded on attempt 2."
    verify_fly_deployment
    return
  fi

  if ! is_transport_failure_log "$LAST_DEPLOY_LOG"; then
    print_error "Deploy failed with non-transport error on attempt 2."
    print_warning "See deploy log: $LAST_DEPLOY_LOG"
    exit 1
  fi

  print_warning "Transport-level builder failure persisted. Retrying in 30s with --wg=false."
  sleep 30

  if run_fly_deploy_attempt "3" --depot=false --recreate-builder --wg=false; then
    print_status "Fly deploy succeeded on attempt 3."
    verify_fly_deployment
    return
  fi

  print_error "Fly deploy failed after transport-retry strategy."
  print_warning "Last deploy log: $LAST_DEPLOY_LOG"
  exit 1
}

trigger_render_deploy() {
  if [[ -z "$RENDER_DEPLOY_HOOK_URL" ]]; then
    print_warning "RENDER_DEPLOY_HOOK_URL is not set. Skipping automated deploy trigger."
    print_status "Trigger deployment from the Render dashboard or push to the tracked branch."
    return
  fi

  print_step "Triggering Render deploy via deploy hook..."
  if curl -fsS -X POST "$RENDER_DEPLOY_HOOK_URL" >/dev/null; then
    print_status "Render deploy hook triggered successfully."
  else
    print_error "Failed to trigger Render deploy hook."
    exit 1
  fi
}

deploy_to_render() {
  print_step "Preparing Render deployment..."
  trigger_render_deploy
  verify_render_deployment
}

get_deployment_url() {
  print_step "Getting deployment URL..."

  case "$PLATFORM" in
    "render")
      print_status "Backend deployed at: ${RENDER_BACKEND_URL%/}"
      ;;
    "fly")
      if ! command -v jq >/dev/null 2>&1; then
        print_warning "jq not found; skipping automated Fly URL lookup."
        return
      fi

      local fly_url
      fly_url="$(fly status --json --app "$APP_NAME" | jq -r '.Hostname // empty')"
      if [[ -n "$fly_url" ]]; then
        print_status "Backend deployed at: https://$fly_url"
      fi
      ;;
  esac
}

main() {
  trap cleanup_staging_context EXIT

  echo "Goblin Assistant Backend Production Deployment"
  echo "=============================================="
  echo "Platform: $PLATFORM"
  echo ""

  cd "$SCRIPT_DIR"

  if [[ ! -f ".env.production" ]]; then
    print_warning ".env.production not found; continuing (runtime secrets may already be set in your deployment platform)."
  fi

  case "$PLATFORM" in
    "render")
      deploy_to_render
      ;;
    "fly")
      deploy_to_fly
      ;;
  esac

  get_deployment_url

  echo ""
  print_status "Backend deployment completed."
  echo ""
  print_status "Next steps:"
  echo "1. Verify service health and logs"
  echo "2. Deploy frontend if needed"
  echo "3. Validate end-to-end requests"
}

PLATFORM="${1:-render}"
case "$PLATFORM" in
  "render"|"fly") ;;
  *)
    print_error "Invalid platform. Use: render (default) or fly (deprecated rollback)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 render"
    echo "  $0 fly"
    exit 1
    ;;
esac

main
