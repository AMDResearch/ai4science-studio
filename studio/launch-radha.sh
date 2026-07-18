#!/bin/bash
# AMD AI4Science Studio — Vultr Lux cluster launcher
# Binds to 0.0.0.0 so ngrok / SSH tunnels can reach both services.
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

# Frontend — expose on all interfaces for ngrok
echo "[studio] Starting frontend on 0.0.0.0:5275..."
cd "$FRONTEND"
if [ ! -d "node_modules" ]; then
  npm install --silent
fi
npm run dev -- --host 0.0.0.0 --port 5275 2>&1 | sed 's/^/[frontend] /' &

echo ""
echo "[studio] ✅ Studio up on Vultr Lux."
echo "[studio]    To share externally:"
echo "[studio]      ngrok http 5275    # single tunnel for the full app"
echo "[studio]    Backend health: http://\$(hostname -f):8275/api/health"
echo "[studio]    Frontend:       http://\$(hostname -f):5275"
echo ""
wait
