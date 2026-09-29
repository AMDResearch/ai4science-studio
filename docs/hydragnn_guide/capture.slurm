#!/bin/bash
#SBATCH --job-name=hg_guide_shots
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:25:00
#SBATCH --output=capture_%j.log
# Submit from docs/hydragnn_guide inside the checkout.

set -uo pipefail
echo "[shots] Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "[shots] ulimit -u: $(ulimit -u)"

REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
STUDIO="${STUDIO_DIR:-$REPO/studio}"
GUIDE="$REPO/docs/hydragnn_guide"
BACK_PORT=8376
FRONT_PORT=5299
export FRONT_URL="http://127.0.0.1:${FRONT_PORT}"
export BACK_URL="http://127.0.0.1:${BACK_PORT}"
export SHOTS_DIR="$GUIDE/shots"
mkdir -p "$SHOTS_DIR"
# Control file: capture_shots.js writes a struct index here; backend reads it to
# pick which curated material the demo returns.
export HG_DEMO_STRUCT_INDEX_FILE="$GUIDE/.demo_struct_index"
echo 0 > "$HG_DEMO_STRUCT_INDEX_FILE"

# GTK/X libs Chromium needs (no root). Compute nodes are air-gapped, so the
# .deb files are pre-staged on the login node (apt-get download ...) into
# $STAGED_DEBS on a shared filesystem.
STAGED_DEBS="${STAGED_DEBS:?set STAGED_DEBS to a shared dir of pre-downloaded GTK/X .deb files}"
DEPS="${AI4S_LOCAL_SCRATCH:-${TMPDIR:-/tmp}/$USER}/hgshots_deps_$$"
CHROME_LOCAL="${AI4S_LOCAL_SCRATCH:-${TMPDIR:-/tmp}/$USER}/hgshots_chrome_$$"
mkdir -p "$DEPS" "$CHROME_LOCAL"
for d in "$STAGED_DEBS"/*.deb; do [ -f "$d" ] && dpkg-deb -x "$d" "$DEPS/" 2>/dev/null; done
echo "[shots] extracted $(ls "$STAGED_DEBS"/*.deb 2>/dev/null | wc -l) staged debs"
export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu:$DEPS/usr/lib:${LD_LIBRARY_PATH:-}"

# Chromium on local scratch (NFS mmap of ICU data fails otherwise)
cp -r ~/.cache/ms-playwright/chromium-1228 "$CHROME_LOCAL/" 2>/dev/null || true
export PLAYWRIGHT_BROWSERS_PATH="$CHROME_LOCAL"

# Backend + Vite (their own instances on this node)
cd "$STUDIO/backend"
.venv/bin/python3 -m uvicorn main:app --host 127.0.0.1 --port $BACK_PORT > "$GUIDE/backend_$SLURM_JOB_ID.log" 2>&1 &
BACK_PID=$!
cd "$STUDIO/frontend"
# vite.config proxies /api -> 127.0.0.1:8376 (matches BACK_PORT above)
npm run dev -- --host 127.0.0.1 --port $FRONT_PORT > "$GUIDE/frontend_$SLURM_JOB_ID.log" 2>&1 &
FRONT_PID=$!
cleanup() { kill $BACK_PID $FRONT_PID 2>/dev/null || true; rm -rf "$DEPS" "$CHROME_LOCAL"; }
trap cleanup EXIT

for i in $(seq 1 30); do curl -sf "$BACK_URL/api/health" >/dev/null 2>&1 && echo "[shots] backend healthy" && break || sleep 2; done
for i in $(seq 1 40); do curl -sf "$FRONT_URL/" >/dev/null 2>&1 && echo "[shots] frontend healthy" && break || sleep 2; done
curl -sf "$BACK_URL/api/health" >/dev/null || { echo "[shots] backend not up"; tail "$GUIDE/backend_$SLURM_JOB_ID.log"; exit 1; }
curl -sf "$FRONT_URL/" >/dev/null || { echo "[shots] frontend not up"; tail -20 "$GUIDE/frontend_$SLURM_JOB_ID.log"; exit 1; }
sleep 6

cd "$GUIDE"
node capture_shots.js
RC=$?
echo "[shots] capture exit $RC"
ls -lh "$SHOTS_DIR"/*.png 2>/dev/null | tail
echo "[shots] done at $(date)"
exit $RC
