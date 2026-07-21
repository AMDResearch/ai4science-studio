#!/bin/bash
# AMD AI4Science Studio — compute-node launcher (verified working 2026-07-21).
#
# HARD-WON LEARNINGS baked into this script (see studio/SERVING.md for the full
# runbook and topology diagram):
#
#   1. Services MUST run on the COMPUTE NODE, attached to a persistent SLURM holder
#      allocation via `srun --jobid=<alloc> --overlap`. Running uvicorn/vite on the
#      LOGIN node makes the vite `/api` proxy target (compute 127.0.0.1:8376)
#      unreachable -> every API call 500s. This bit us; do not "just run uvicorn".
#   2. Ports are FIXED at 8376 (backend) / 5376 (frontend) because the vite proxy
#      hard-codes `/api -> 127.0.0.1:8376` (frontend/vite.config.js) and the
#      Cloudflare tunnel config points at 5376. Change here => change both of those.
#   3. Backend runs with --reload so edits to backend/*.py (jobs.py routing,
#      prompts.py enable matrix, main.py guards) apply without a manual restart.
#      Frontend has HMR (CHOKIDAR_USEPOLLING=true because NFS breaks inotify).
#   4. Stopping is fiddly: `npm run dev` spawns a child vite that OUTLIVES the srun
#      wrapper, and `scancel -n` / `--jobid` step-targeting is unsupported here.
#      We kill login-node wrappers with `pkill -f "job-name=..."` AND kill the
#      surviving compute-node listeners by PID over an `srun --overlap` shell.
#   5. Compute nodes are air-gapped (no internet). Never pip/npm install here — do
#      that on the login node first. This script assumes .venv and node_modules
#      already exist.
#   6. The laptop can reach ONLY the login node, NEVER compute nodes. So access is
#      TWO hops: laptop -> login node (ssh -L :5376) -> compute node (a login-node
#      ssh forward this script creates). The laptop tunnels to login-node
#      `localhost:5376`, NOT the compute hostname (which it cannot resolve/reach).
#   7. Cloudflare (public URL) runs on the login node and its ingress `service:`
#      hardcodes the compute-node hostname. When the studio moves nodes, update
#      /shared/spannala/.cloudflared/config.yml and restart cloudflared (SERVING.md).
#
# Usage:
#   ./launch-local.sh                 # auto-detect the holder allocation
#   STUDIO_JOBID=17694 ./launch-local.sh   # or pin a specific allocation job id

set -uo pipefail
STUDIO="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$STUDIO/backend"
FRONTEND="$STUDIO/frontend"
mkdir -p "$STUDIO/logs" "$BACKEND/logs" "$FRONTEND/logs"

BACKEND_PORT=8376
FRONTEND_PORT=5376

# ── 1. Resolve the compute-node holder allocation ────────────────────────────
# Prefer an explicit STUDIO_JOBID; else the newest RUNNING "studio-hold" job;
# else the newest RUNNING plain "bash" holder (the legacy interactive allocation).
resolve_alloc() {
  if [[ -n "${STUDIO_JOBID:-}" ]]; then echo "$STUDIO_JOBID"; return; fi
  local j
  j=$(squeue -u "$USER" -h -t RUNNING -o "%i %j %M" 2>/dev/null \
        | awk '$2=="studio-hold"{print $1}' | sort -n | tail -1)
  [[ -z "$j" ]] && j=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" 2>/dev/null \
        | awk '$2=="bash"{print $1}' | sort -n | tail -1)
  echo "$j"
}
JOBID="$(resolve_alloc)"
if [[ -z "$JOBID" ]]; then
  echo "[studio] ERROR: no running holder allocation found."
  echo "[studio] Submit one first:  sbatch scripts/launch/studio-hold-4d.sbatch"
  echo "[studio] Then re-run, or pass STUDIO_JOBID=<id> ./launch-local.sh"
  exit 1
fi
NODE="$(squeue -j "$JOBID" -h -o '%N' 2>/dev/null)"
echo "[studio] Using allocation job ${JOBID} on node ${NODE:-?}"
SRUN="srun --jobid=${JOBID} --overlap"

# ── 2. Stop any stale services (wrappers + surviving compute-node listeners) ──
echo "[studio] Stopping any existing backend/frontend..."
pkill -f "job-name=hg_backend"  2>/dev/null || true
pkill -f "job-name=hg_frontend" 2>/dev/null || true
sleep 2
# npm/vite child survives the wrapper: kill whatever still holds our ports on the
# compute node, by PID, so the relaunch binds cleanly.
$SRUN bash -c '
  for p in 8376 5376; do
    pid=$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done' 2>/dev/null || true
sleep 2

# ── 3. Launch backend on the compute node (uvicorn :8376, --reload) ──────────
echo "[studio] Starting backend on ${NODE}:${BACKEND_PORT} (--reload)..."
nohup $SRUN --job-name=hg_backend bash -c \
  "cd '$BACKEND' && HG_USE_PBC_EDGES=1 exec .venv/bin/python3 .venv/bin/uvicorn \
   main:app --host 0.0.0.0 --port ${BACKEND_PORT} --reload" \
  > "$BACKEND/logs/backend_restart.log" 2>&1 &

# ── 4. Launch frontend on the compute node (vite :5376, polling HMR) ─────────
echo "[studio] Starting frontend on ${NODE}:${FRONTEND_PORT}..."
nohup $SRUN --job-name=hg_frontend bash -c \
  "cd '$FRONTEND' && CHOKIDAR_USEPOLLING=true exec npm run dev -- \
   --host 0.0.0.0 --port ${FRONTEND_PORT}" \
  > "$FRONTEND/logs/frontend_restart.log" 2>&1 &

# ── 5. Health check ──────────────────────────────────────────────────────────
echo "[studio] Waiting for services..."
sleep 10
$SRUN bash -c "
  curl -s -m5 -o /dev/null -w '[studio] backend  %{http_code} (expect 200)\n' http://127.0.0.1:${BACKEND_PORT}/api/health
  curl -s -m5 -o /dev/null -w '[studio] frontend %{http_code} (expect 200)\n' http://127.0.0.1:${FRONTEND_PORT}/
" 2>/dev/null | grep -E "backend|frontend"

# ── 6. Login-node -> compute forward (hop 1 of the laptop's 2-hop tunnel) ─────
# The laptop can reach ONLY the login node, never compute nodes. So the LOGIN node
# must forward its own 127.0.0.1:${FRONTEND_PORT} to the compute node's frontend.
# The laptop then does `ssh -L ${FRONTEND_PORT}:localhost:${FRONTEND_PORT}
# rad-vultr-login` (targeting login-node localhost, NOT the compute hostname the
# laptop can't resolve). We (re)create this forward here so it always tracks ${NODE}.
echo "[studio] (Re)creating login->compute forward for 127.0.0.1:${FRONTEND_PORT} -> ${NODE}..."
# Kill any stale forward bound to our port (e.g. pointing at a previous node).
stale=$(ss -ltnp 2>/dev/null | grep ":${FRONTEND_PORT} " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
[ -n "$stale" ] && kill "$stale" 2>/dev/null || true
sleep 1
nohup ssh -N -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
  -L "127.0.0.1:${FRONTEND_PORT}:localhost:${FRONTEND_PORT}" "${NODE}" \
  > "$STUDIO/logs/headnode_tunnel_${NODE}.log" 2>&1 &
sleep 3
curl -s -m5 -o /dev/null -w "[studio] login localhost:${FRONTEND_PORT} -> %{http_code} (expect 200)\n" \
  "http://127.0.0.1:${FRONTEND_PORT}/" || true

echo ""
echo "[studio] ✅ Studio up on node ${NODE} (job ${JOBID})."
echo "[studio]    Path: laptop --ssh--> login node --ssh forward--> ${NODE} (compute)."
echo "[studio]    The laptop reaches ONLY the login node, so tunnel to login-node"
echo "[studio]    localhost (NOT ${NODE}, which the laptop can't reach). The login->"
echo "[studio]    compute forward above bridges the last hop."
echo "[studio]    On your laptop:"
echo "[studio]      ssh -N -L ${FRONTEND_PORT}:localhost:${FRONTEND_PORT} rad-vultr-login"
echo "[studio]      http://localhost:${FRONTEND_PORT}"
echo "[studio]    Public URL (Cloudflare): https://ai4science-studio.axiomfabric.ai"
echo "[studio]      (if the node changed, update config.yml ingress to ${NODE} and"
echo "[studio]       restart cloudflared — see studio/SERVING.md)."
echo "[studio]    Logs: $BACKEND/logs/backend_restart.log , $FRONTEND/logs/frontend_restart.log"
echo "[studio]    Backend edits (*.py) auto-reload; frontend uses HMR."
echo "[studio] This launcher returns; services keep running on the allocation."
