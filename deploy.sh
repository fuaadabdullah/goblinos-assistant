#!/bin/bash

# Goblin Assistant Deployment Script
# Primary backend target: Render
# Fly.io backend deployment is retained for rollback only.

set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
APP_DIR="$REPO_ROOT/apps/goblin-assistant"
BACKEND_PLATFORM="${BACKEND_PLATFORM:-render}"
case "$BACKEND_PLATFORM" in
  "render"|"fly") ;;
  *)
    echo "Invalid BACKEND_PLATFORM: $BACKEND_PLATFORM (expected: render or fly)"
    exit 1
    ;;
esac

echo "🚀 Goblin Assistant Deployment Script"
echo "===================================="
echo "Repo Root: $REPO_ROOT"
echo "App Dir: $APP_DIR"
echo "Backend Platform: $BACKEND_PLATFORM"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse command line arguments
DEPLOY_BACKEND=${1:-"both"}
DEPLOY_FRONTEND=${2:-"both"}

case "$DEPLOY_BACKEND" in
  "backend"|"both"|"all")
    DO_BACKEND=true
    ;;
  *)
    DO_BACKEND=false
    ;;
esac

case "$DEPLOY_FRONTEND" in
  "frontend"|"both"|"vercel"|"all")
    DO_FRONTEND=true
    ;;
  *)
    DO_FRONTEND=false
    ;;
esac

# Step 1: Pre-deployment checks
echo -e "${BLUE}Step 1: Running pre-deployment checks...${NC}"
cd "$APP_DIR"

if [[ -f "pre-deploy.sh" ]]; then
  bash pre-deploy.sh
else
  echo -e "${YELLOW}Warning: pre-deploy.sh not found${NC}"
fi

# Step 2: Deploy backend
if [[ "$DO_BACKEND" == true ]]; then
  echo ""
  if [[ "$BACKEND_PLATFORM" == "fly" ]]; then
    echo -e "${YELLOW}Step 2: Deploying backend to Fly.io (DEPRECATED rollback path)...${NC}"
  else
    echo -e "${BLUE}Step 2: Deploying backend to Render...${NC}"
  fi

  if [[ ! -x "./deploy-backend.sh" ]]; then
    echo -e "${RED}❌ deploy-backend.sh not found or not executable${NC}"
    exit 1
  fi

  ./deploy-backend.sh "$BACKEND_PLATFORM"
  echo -e "${GREEN}✓ Backend deployment step completed${NC}"
fi

# Step 3: Deploy frontend to Vercel
if [[ "$DO_FRONTEND" == true ]]; then
  echo ""
  echo -e "${BLUE}Step 3: Deploying frontend to Vercel...${NC}"

  if command -v vercel &> /dev/null; then
    echo -e "${YELLOW}Deploying to Vercel...${NC}"
    vercel deploy --prod

    echo -e "${GREEN}✓ Frontend deployed successfully${NC}"
    echo ""
    echo "Visit your deployment:"
    echo "  https://goblin-assistant.vercel.app"
  else
    echo -e "${YELLOW}Vercel CLI not found, assuming Git deployment...${NC}"
    echo ""
    echo "To deploy via Vercel Git integration:"
    echo "  git push origin main"
    echo ""
    echo "Or install Vercel CLI:"
    echo "  npm i -g vercel"
  fi
fi

# Step 4: Post-deployment verification
echo ""
echo -e "${BLUE}Step 4: Post-deployment verification...${NC}"

BACKEND_URL="${RENDER_BACKEND_URL:-https://goblin-backend.onrender.com}"
if [[ "$BACKEND_PLATFORM" == "fly" ]]; then
  BACKEND_URL="${FLY_BACKEND_URL:-https://goblin-backend.fly.dev}"
fi

echo ""
echo "Checking API endpoints:"
echo "  Frontend: https://goblin-assistant.vercel.app"
echo "  Backend: ${BACKEND_URL}"
echo "  Health: ${BACKEND_URL%/}/health"

# Step 5: Monitor logs
echo ""
echo -e "${BLUE}Step 5: Monitoring deployment...${NC}"

if [[ "$DO_BACKEND" == true ]]; then
  if [[ "$BACKEND_PLATFORM" == "fly" ]] && command -v fly &> /dev/null; then
    echo ""
    echo -e "${YELLOW}Recent Fly backend logs (last 20 lines):${NC}"
    fly logs --app goblin-backend --lines 20 || true
  elif [[ "$BACKEND_PLATFORM" == "render" ]]; then
    echo ""
    echo "Render logs are available in the Render dashboard for service: goblin-backend"
  fi
fi

echo ""
echo "=================================================="
echo -e "${GREEN}✅ Deployment process complete!${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Visit https://goblin-assistant.vercel.app"
echo "2. Verify chat functionality"
echo "3. Check error logs in Sentry"
echo "4. Monitor performance in Datadog"
echo ""
echo "Troubleshooting:"
echo "  Render logs: https://dashboard.render.com"
echo "  Vercel logs: vercel logs"
echo "  Fly rollback (deprecated): BACKEND_PLATFORM=fly ./deploy-backend.sh fly"
echo ""
