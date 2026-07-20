#!/bin/bash
#SBATCH --job-name=record-demos
#SBATCH --partition=lux
#SBATCH --account=vultr_lux
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:15:00
#SBATCH --output=/home/spannala/Projects/ai4science-studio/studio/scripts/demo/record_demos_%j.log

set -e
echo "[demos] Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "[demos] ulimit -u: $(ulimit -u)"

STUDIO="$HOME/Projects/ai4science-studio/studio"
BACK_PORT=8299
FRONT_PORT=5299
export FRONT_URL="http://127.0.0.1:${FRONT_PORT}"
export BACK_URL="http://127.0.0.1:${BACK_PORT}"
export OUT_DIR="$STUDIO/demo/demo-output"
mkdir -p "$OUT_DIR"

# Stage GTK/X libs Chromium needs
DEPS="/scratch/$USER/rdemos_deps_$$"
CHROME_LOCAL="/scratch/$USER/rdemos_chrome_$$"
mkdir -p "$DEPS" "$CHROME_LOCAL"
cd /tmp
apt-get download libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libasound2t64 \
  libxcomposite1 libxdamage1 libxrandr2 libatspi2.0-0t64 libgbm1 libxkbcommon0 libxi6 \
  2>&1 | grep -vE "^Get:|^Fetched|^Reading|^W:" || true
for d in /tmp/*.deb; do [ -f "$d" ] && dpkg-deb -x "$d" "$DEPS/" 2>/dev/null; done
export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu:$DEPS/usr/lib:${LD_LIBRARY_PATH:-}"
cp -r ~/.cache/ms-playwright/chromium-1228 "$CHROME_LOCAL/"
export PLAYWRIGHT_BROWSERS_PATH="$CHROME_LOCAL"

# Find ffmpeg
FFMPEG_CANDIDATES=(
  "$STUDIO/backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
  "$HOME/Projects/Utils/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
for f in "${FFMPEG_CANDIDATES[@]}"; do
  if [ -f "$f" ]; then export FFMPEG="$f"; echo "[demos] FFMPEG=$FFMPEG"; break; fi
done

cleanup() { kill $BACK_PID $FRONT_PID 2>/dev/null||true; rm -rf "$DEPS" "$CHROME_LOCAL"; }
trap cleanup EXIT

# Start backend
cd "$STUDIO/backend"
.venv/bin/python3 -m uvicorn main:app --host 127.0.0.1 --port $BACK_PORT > "$OUT_DIR/backend_$SLURM_JOB_ID.log" 2>&1 &
BACK_PID=$!
echo "[demos] backend pid $BACK_PID"

# Start Vite frontend
cd "$STUDIO/frontend"
npm run dev -- --host 127.0.0.1 --port $FRONT_PORT > "$OUT_DIR/frontend_$SLURM_JOB_ID.log" 2>&1 &
FRONT_PID=$!
echo "[demos] frontend pid $FRONT_PID"

# Wait for services
for i in $(seq 1 30); do
  curl -sf "$BACK_URL/api/health" >/dev/null 2>&1 && echo "[demos] backend healthy" && break || sleep 2
done
for i in $(seq 1 40); do
  curl -sf "$FRONT_URL/" >/dev/null 2>&1 && echo "[demos] frontend healthy" && break || sleep 2
done
sleep 6  # Vite dep pre-bundling

echo "[demos] Recording HydraGNN demo ..."
cd "$STUDIO/demo"
node "$STUDIO/scripts/demo/record_hydragnn.js"
echo "[demos] HydraGNN done"

echo "[demos] Recording ORBIT-2 DC demo ..."
node "$STUDIO/scripts/demo/record_orbit2_dc.js"
echo "[demos] ORBIT-2 done"

echo "[demos] Recording GP-MoLFormer demo ..."
node "$STUDIO/scripts/demo/record_gpmolformer.js"
echo "[demos] GP-MoLFormer done"

echo ""
echo "[demos] All recordings complete at $(date)"
ls -lh "$OUT_DIR"/*.mp4 2>/dev/null
