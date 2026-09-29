#!/bin/bash
# AMD AI4Science Studio — cluster login-node launcher
# Binds to 0.0.0.0 so the Cloudflare tunnel / SSH tunnels can reach both services.
# Run this on the cluster login node.

set -e
STUDIO="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$STUDIO/backend"
FRONTEND="$STUDIO/frontend"

cleanup() { kill 0; }
trap cleanup INT TERM EXIT

# Backend
echo "[studio] Starting backend on 0.0.0.0:8275..."
cd "$BACKEND"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8275 --reload 2>&1 | sed 's/^/[backend] /' &

# Frontend — expose on all interfaces for the Cloudflare tunnel
echo "[studio] Starting frontend on 0.0.0.0:5275..."
cd "$FRONTEND"
if [ ! -d "node_modules" ]; then
  npm install --silent
fi
npm run dev -- --host 0.0.0.0 --port 5275 2>&1 | sed 's/^/[frontend] /' &

echo ""
echo "[studio] ✅ Studio up on $(hostname -s)."
echo "[studio]    To share externally (Cloudflare named tunnel):"
echo "[studio]      cloudflared tunnel --config \${CLOUDFLARED_CONFIG:-<your config.yml>} run <tunnel-name>"
echo "[studio]      Public URL: ${STUDIO_PUBLIC_URL:-<your tunnel hostname>}"
echo "[studio]    Backend health: http://\$(hostname -f):8275/api/health"
echo "[studio]    Frontend:       http://\$(hostname -f):5275"
echo ""
wait
