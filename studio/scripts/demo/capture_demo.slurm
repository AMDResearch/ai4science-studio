#!/bin/bash
#SBATCH --job-name=hg_demo_pdf
#SBATCH --partition=lux
#SBATCH --account=vultr_lux
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:25:00
#SBATCH --output=/home/spannala/Projects/ai4science-studio/studio/scripts/demo/logs/capture_demo_%j.log

set -uo pipefail
echo "[demo] Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "[demo] ulimit -u: $(ulimit -u)"

STUDIO="$HOME/Projects/ai4science-studio/studio"
BACK_PORT=8299
FRONT_PORT=5299
export FRONT_URL="http://127.0.0.1:${FRONT_PORT}"
export BACK_URL="http://127.0.0.1:${BACK_PORT}"
export OUT_DIR="$STUDIO/demo/demo-output"
mkdir -p "$OUT_DIR/logs"

# ── Stage GTK/X libs Chromium needs (no root) ───────────────────────────────
DEPS="/scratch/$USER/hgdemo_deps_$$"
CHROME_LOCAL="/scratch/$USER/hgdemo_chrome_$$"
mkdir -p "$DEPS" "$CHROME_LOCAL"
cd /tmp
apt-get download \
  libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libasound2t64 \
  libxcomposite1 libxdamage1 libxrandr2 libatspi2.0-0t64 \
  libgbm1 libxkbcommon0 libxi6 2>&1 | grep -vE "^Get:|^Fetched|^Reading|^W:" || true
for d in /tmp/*.deb; do [ -f "$d" ] && dpkg-deb -x "$d" "$DEPS/" 2>/dev/null; done
export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu:$DEPS/usr/lib:${LD_LIBRARY_PATH:-}"

# Chromium on local scratch (NFS mmap issues otherwise)
cp -r ~/.cache/ms-playwright/chromium-1228 "$CHROME_LOCAL/" 2>/dev/null || true
export PLAYWRIGHT_BROWSERS_PATH="$CHROME_LOCAL"

# ── Start backend ───────────────────────────────────────────────────────────
cd "$STUDIO/backend"
.venv/bin/python3 -m uvicorn main:app --host 127.0.0.1 --port $BACK_PORT > "$OUT_DIR/logs/backend_$SLURM_JOB_ID.log" 2>&1 &
BACK_PID=$!
echo "[demo] backend pid $BACK_PID"

# ── Start Vite dev server ───────────────────────────────────────────────────
cd "$STUDIO/frontend"
npm run dev -- --host 127.0.0.1 --port $FRONT_PORT > "$OUT_DIR/logs/frontend_$SLURM_JOB_ID.log" 2>&1 &
FRONT_PID=$!
echo "[demo] frontend pid $FRONT_PID"

cleanup() { kill $BACK_PID $FRONT_PID 2>/dev/null || true; rm -rf "$DEPS" "$CHROME_LOCAL"; }
trap cleanup EXIT

# ── Wait for health ─────────────────────────────────────────────────────────
for i in $(seq 1 30); do
  curl -sf "$BACK_URL/api/health" >/dev/null 2>&1 && echo "[demo] backend healthy" && break || sleep 2
done
for i in $(seq 1 40); do
  curl -sf "$FRONT_URL/" >/dev/null 2>&1 && echo "[demo] frontend healthy" && break || sleep 2
done
curl -sf "$BACK_URL/api/health" >/dev/null || { echo "[demo] backend not up"; cat "$OUT_DIR/logs/backend_$SLURM_JOB_ID.log" | tail; exit 1; }
curl -sf "$FRONT_URL/" >/dev/null || { echo "[demo] frontend not up"; tail -20 "$OUT_DIR/logs/frontend_$SLURM_JOB_ID.log"; exit 1; }

# Give Vite a moment to finish dep pre-bundling
sleep 5

# ── Capture + build PDF ─────────────────────────────────────────────────────
cd "$STUDIO/demo"
node "$STUDIO/scripts/demo/capture_hydragnn_demo.js"
RC=$?
echo "[demo] capture exit $RC"
ls -lh "$OUT_DIR/hydragnn_demo_walkthrough.pdf" 2>/dev/null || echo "[demo] no PDF produced"
echo "[demo] done at $(date)"
exit $RC
