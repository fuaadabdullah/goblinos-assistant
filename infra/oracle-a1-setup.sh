#!/bin/bash
# Goblin llama.cpp setup for an Oracle A1 VM (ARM64).
# Installs llama.cpp, downloads the uncensored goblin-core model,
# and starts it as a systemd service behind an API key.
# Run ON the VM as root:  curl -fsSL <raw-url> | sudo bash
# At the end it prints the ENDPOINT and API KEY -- send both to Goblin.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "[goblin] installing packages..."
apt-get update -qq
apt-get install -y -qq curl ca-certificates unzip python3 > /dev/null

echo "[goblin] installing llama.cpp (ARM64)..."
cd /tmp
TAG=$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")
ZIP="llama-${TAG#?}-bin-ubuntu-arm64.zip"
for i in 1 2 3; do
  curl -fsSL -o llama.zip "https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/${ZIP}" && break
  echo "[goblin] llama.cpp download retry $i"; sleep 10
done
rm -rf /opt/llama.cpp && mkdir -p /opt/llama.cpp
unzip -o -q llama.zip -d /opt/llama.cpp
ln -sf /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
llama-server --version 2>&1 | head -1

echo "[goblin] downloading uncensored model (Qwen3-4B-abliterated, ~2.5GB)..."
mkdir -p /opt/goblin/models /etc/goblin
cd /opt/goblin/models
for i in 1 2 3 4 5; do
  curl -fsSL --retry 3 -o goblin-core.gguf \
    "https://huggingface.co/Mungert/Qwen3-4B-abliterated-GGUF/resolve/main/Qwen3-4B-abliterated-q4_k_m.gguf?download=true" && break
  echo "[goblin] model download retry $i"; sleep 15
done
ls -la goblin-core.gguf

echo "[goblin] generating API key..."
[ -f /etc/goblin/llama-api-keys ] || head -c 32 /dev/urandom | base64 | tr -d '\n' > /etc/goblin/llama-api-keys
chmod 600 /etc/goblin/llama-api-keys

# Bind to the Tailscale IP when Tailscale is up, else loopback only.
TSIP=$(tailscale ip -4 2>/dev/null | head -1 || true)
HOST="127.0.0.1"
[ -n "$TSIP" ] && HOST="$TSIP"
echo "[goblin] binding llama-server to $HOST:8081"

cat > /etc/systemd/system/goblin-llama.service <<EOF
[Unit]
Description=Goblin llama.cpp router
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/llama-server --model /opt/goblin/models/goblin-core.gguf --alias goblin-core --ctx-size 4096 --threads 1 --threads-batch 1 --host $HOST --port 8081 --api-key-file /etc/goblin/llama-api-keys
Restart=always
RestartSec=5
Nice=-5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now goblin-llama.service
sleep 8
if curl -fsS -m 20 -H "Authorization: Bearer $(cat /etc/goblin/llama-api-keys)" "http://$HOST:8081/health"; then
  echo ""
  echo "[goblin] llama-server healthy"
else
  echo "[goblin] WARNING: health check failed -- see: journalctl -u goblin-llama.service"
fi

echo ""
echo "==================== GOBLIN LLAMA READY ===================="
echo "ENDPOINT: http://$HOST:8081"
echo "API KEY: $(cat /etc/goblin/llama-api-keys)"
echo "============================================================"
echo "Send the ENDPOINT and API KEY to Goblin to wire up the provider."
