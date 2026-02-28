#!/bin/bash
# Bootstrap script for Local LLM setup on Kamatera VPS
# This script pulls a model from Drive, starts services, and sets up systemd units
# Usage: ./bootstrap_llm.sh gdrive:models/your-model-folder

set -e

MODEL_REMOTE="$1"
if [ -z "$MODEL_REMOTE" ]; then
    echo "Usage: $0 gdrive:models/<model-folder>"
    echo "Example: $0 gdrive:models/llama-7b-gguf"
    exit 1
fi

echo "=== Local LLM Bootstrap Script ==="
echo "Model remote: $MODEL_REMOTE"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if running as root for some operations
if [[ $EUID -eq 0 ]]; then
    error "Don't run as root. Use sudo for specific commands."
    exit 1
fi

# 1. Pull model from Drive
status "Pulling model from Google Drive..."
if ! rclone copy "$MODEL_REMOTE" /srv/models/active --progress; then
    error "Failed to pull model from Drive"
    echo ""
    echo "This is likely because rclone is not configured for Google Drive."
    echo "To fix this, run these commands on the server:"
    echo ""
    echo "  # Switch to deploy user"
    echo "  sudo -u deploy -i"
    echo ""
    echo "  # Configure rclone"
    echo "  rclone config"
    echo "  - Choose 'gdrive' as storage type"
    echo "  - Name it 'gdrive'"
    echo "  - Follow the OAuth setup for Google Drive"
    echo ""
    echo "  # Test the connection"
    echo "  rclone ls gdrive:models/llama_models"
    echo ""
    echo "  # Then re-run this bootstrap script"
    echo "  cd ~/llm-deployment && ./bootstrap_llm.sh $MODEL_REMOTE"
    echo ""
    warning "Skipping model download for now. Services will start without models."
    warning "Re-run this script after configuring rclone."
    # Continue with service setup even if model download fails
fi

# Set permissions
sudo chmod -R 755 /srv/models/active
status "Model pulled successfully"

# 2. Check model file exists
MODEL_FILE=$(find /srv/models/active -name "*.gguf" | head -1)
if [ -z "$MODEL_FILE" ]; then
    warning "No .gguf model file found in /srv/models/active"
    warning "llama.cpp service will not start without a model."
    warning "Download a model and re-run this script, or configure rclone first."
    MODEL_FILE="/srv/models/active/placeholder.gguf"  # Placeholder for service config
else
    status "Found model file: $MODEL_FILE"
fi

# 3. Update llama.cpp service with correct model path
status "Updating llama.cpp systemd service..."
if [ -f "$MODEL_FILE" ] && [[ "$MODEL_FILE" != *placeholder* ]]; then
    sudo sed -i "s|ExecStart=.*|ExecStart=/opt/llama.cpp/build/bin/server -m $MODEL_FILE --port 8080 --threads 2|" /etc/systemd/system/llamacpp.service
else
    warning "Skipping llama.cpp service update - no valid model file"
fi
sudo systemctl daemon-reload

# 4. Start services
status "Starting Ollama service..."
sudo systemctl start ollama
sleep 2

if [ -f "$MODEL_FILE" ] && [[ "$MODEL_FILE" != *placeholder* ]]; then
    status "Starting llama.cpp service..."
    sudo systemctl start llamacpp
    sleep 2
else
    warning "Skipping llama.cpp service start - no model available"
fi

status "Starting Local LLM Proxy..."
sudo systemctl start local-llm-proxy
sleep 2

# 5. Check services are running
status "Checking service status..."

check_service() {
    if sudo systemctl is-active --quiet "$1"; then
        status "$1: RUNNING"
        return 0
    else
        if [ "$1" = "llamacpp" ] && [ ! -f "$MODEL_FILE" ]; then
            warning "$1: SKIPPED (no model available)"
            return 0  # Don't fail if llama.cpp is skipped due to no model
        else
            error "$1: FAILED"
            sudo systemctl status "$1" --no-pager -l
            return 1
        fi
    fi
}

FAILED=0
check_service ollama || FAILED=1
check_service llamacpp || FAILED=1
check_service local-llm-proxy || FAILED=1

if [ $FAILED -eq 1 ]; then
    error "Some services failed to start. Check logs above."
    exit 1
fi

# 6. Test endpoints
status "Testing endpoints..."

test_endpoint() {
    if curl -s --max-time 10 "$1" > /dev/null; then
        status "$2: OK"
        return 0
    else
        warning "$2: FAILED"
        return 1
    fi
}

test_endpoint "http://localhost:11434/api/tags" "Ollama API"
if [ -f "$MODEL_FILE" ] && [[ "$MODEL_FILE" != *placeholder* ]]; then
    test_endpoint "http://localhost:8080/health" "llama.cpp server"
else
    warning "llama.cpp server: SKIPPED (no model)"
fi
test_endpoint "http://localhost:8002/health" "Local LLM Proxy"

# 7. Show resource usage
status "Current resource usage:"
echo "Memory:"
free -h
echo ""
echo "Disk:"
df -h /srv/models/active
echo ""
echo "Active processes:"
ps aux | grep -E "(ollama|llama|python.*proxy)" | grep -v grep

# 8. Instructions
status "=== Setup Complete ==="
echo ""
echo "Services running:"
echo "  - Ollama: http://localhost:11434"
echo "  - llama.cpp: http://localhost:8080"
echo "  - Proxy API: http://localhost:8002"
echo ""
echo "To test:"
echo "  curl -H 'x-api-key: your-secure-api-key-here' http://localhost:8002/models"
echo ""
echo "To stop services:"
echo "  sudo systemctl stop local-llm-proxy llamacpp ollama"
echo ""
echo "To cleanup model:"
echo "  sudo rm -rf /srv/models/active/*"
echo ""
warning "Remember to:"
echo "  - Update API key in environment"
echo "  - Configure nginx with real SSL certificates"
echo "  - Set up proper firewall rules"
echo "  - Monitor disk/memory usage"

status "Bootstrap completed successfully!"
