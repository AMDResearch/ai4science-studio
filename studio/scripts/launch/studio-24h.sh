#!/bin/bash
STUDIO="${STUDIO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOGS="$STUDIO/logs"; mkdir -p "$LOGS"
echo "NODE: $(hostname) starting studio 24h at $(date)"
cd "$STUDIO/backend"
setsid nohup .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8275 > "$LOGS/studio-backend-24h.log" 2>&1 < /dev/null &
echo "backend PID $!"
cd "$STUDIO/frontend"
setsid nohup npm run dev -- --host 0.0.0.0 --port 5275 > "$LOGS/studio-frontend-24h.log" 2>&1 < /dev/null &
echo "frontend PID $!"
sleep 12
curl -s http://localhost:8275/api/health && echo ""
curl -s -o /dev/null -w "frontend HTTP_%{http_code}\n" http://localhost:5275/
sleep 86400
