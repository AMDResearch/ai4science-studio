#!/bin/bash
# AMD AI4Science Studio — local launcher
# Starts FastAPI backend (port 8099) + Vite frontend (port 5173)

set -e
STUDIO="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$STUDIO/backend"
FRONTEND="$STUDIO/frontend"

cleanup() { kill 0; }
trap cleanup INT TERM EXIT

echo "[studio] Starting backend..."
cd "$BACKEND"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/uvicorn main:app --port 8275 --reload 2>&1 | sed 's/^/[backend] /' &

echo "[studio] Starting frontend..."
cd "$FRONTEND"
if [ ! -d "node_modules" ]; then
  npm install --silent
fi
npm run dev -- --host 127.0.0.1 --port 5275 2>&1 | sed 's/^/[frontend] /' &

echo ""
echo "[studio] ✅ Studio starting up..."
echo "[studio]    Backend:  http://localhost:8275/api/health"
echo "[studio]    Frontend: http://localhost:5275"
echo "[studio] Press Ctrl+C to stop."
echo ""
wait
