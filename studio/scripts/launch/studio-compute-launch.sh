#!/bin/bash
STUDIO="${STUDIO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOGS="$STUDIO/logs"; mkdir -p "$LOGS"
echo "COMPUTE_NODE: $(hostname)"
echo "ulimit -u: $(ulimit -u)"

cd "$STUDIO/backend"
nohup .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8275 > "$LOGS/studio-backend-compute.log" 2>&1 &
echo "backend PID: $!"

cd "$STUDIO/frontend"
nohup npm run dev -- --host 0.0.0.0 --port 5275 > "$LOGS/studio-frontend-compute.log" 2>&1 &
echo "frontend PID: $!"

sleep 10
echo "=== backend health ==="
curl -s http://127.0.0.1:8275/api/health
echo ""
echo "=== frontend ==="
curl -s http://127.0.0.1:5275 | head -3
echo "=== ready, keeping alive ==="
# Keep the srun alive so services persist
sleep 3300
