#!/bin/bash
# Record the three 4K (3840x2160, 16:9) _v1 demo videos.
#
# Run this ON a4, where the studio backend + frontend are ALREADY running.
# It does NOT start or stop any service — it only connects, records, and copies
# the results to ~/transfer.
#
# Typically launched on a4 via the hold job:
#   srun --jobid=<hold-jobid> --overlap bash \
#     ~/Projects/ai4science-studio/studio/scripts/demo/record_demos_v1.sh
#
# Ports default to the fixed studio ports from studio/SERVING.md (frontend 5376,
# backend 8376). Override with FRONT_URL / BACK_URL if the services moved.
#
# Output: ~/transfer/{hydragnn_demo_v1,orbit2_dc_demo_v1,gpmolformer_finetune_demo_v1}.mp4

set -e
echo "[demos_v1] Host $(hostname) at $(date)"

STUDIO="$HOME/Projects/ai4science-studio/studio"
SCRIPTS="$STUDIO/scripts/demo"
# Must match the live a4 services. The frontend's Vite /api proxy must point at
# the backend port, or rendered pages 500 (recorders' own API calls would still
# work, silently producing broken videos). Keep these in sync with the running services.
export FRONT_URL="${FRONT_URL:-http://127.0.0.1:5376}"
export BACK_URL="${BACK_URL:-http://127.0.0.1:8376}"
export OUT_DIR="${OUT_DIR:-$STUDIO/demo/demo-output}"
TRANSFER="${TRANSFER_DIR:-$HOME/transfer}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$TRANSFER"

# ── Stage GTK/X libs Chromium needs + copy Chromium to scratch (compute nodes
#    lack these; Ubuntu 24.04 t64 package suffixes). Same recipe as record_demos.slurm.
DEPS="/scratch/$USER/rdemos_v1_deps_$$"
CHROME_LOCAL="/scratch/$USER/rdemos_v1_chrome_$$"
mkdir -p "$DEPS" "$CHROME_LOCAL"
cleanup() { rm -rf "$DEPS" "$CHROME_LOCAL"; }
trap cleanup EXIT

echo "[demos_v1] Staging GTK/X libs..."
cd /tmp
apt-get download libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libasound2t64 \
  libxcomposite1 libxdamage1 libxrandr2 libatspi2.0-0t64 libgbm1 libxkbcommon0 libxi6 \
  2>&1 | grep -vE "^Get:|^Fetched|^Reading|^W:" || true
for d in /tmp/*.deb; do [ -f "$d" ] && dpkg-deb -x "$d" "$DEPS/" 2>/dev/null; done
export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu:$DEPS/usr/lib:${LD_LIBRARY_PATH:-}"
cp -r ~/.cache/ms-playwright/chromium-1228 "$CHROME_LOCAL/"
export PLAYWRIGHT_BROWSERS_PATH="$CHROME_LOCAL"

# ── Locate ffmpeg (imageio-ffmpeg bundled binary)
FFMPEG_CANDIDATES=(
  "$STUDIO/backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
  "$HOME/Projects/Utils/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
for f in "${FFMPEG_CANDIDATES[@]}"; do
  if [ -f "$f" ]; then export FFMPEG="$f"; echo "[demos_v1] FFMPEG=$FFMPEG"; break; fi
done

# ── Health-check the ALREADY-RUNNING services (do not start them) ──────────────
curl -sf "$BACK_URL/api/health" >/dev/null 2>&1 || {
  echo "[demos_v1] Backend not reachable at $BACK_URL — start the a4 services first."; exit 1
}
echo "[demos_v1] backend healthy at $BACK_URL"
curl -sf "$FRONT_URL/" >/dev/null 2>&1 || {
  echo "[demos_v1] Frontend not reachable at $FRONT_URL — start the a4 services first."; exit 1
}
echo "[demos_v1] frontend healthy at $FRONT_URL"

# ── Record ────────────────────────────────────────────────────────────────────
cd "$STUDIO/demo"

echo "[demos_v1] Recording HydraGNN 4K demo ..."
node "$SCRIPTS/record_hydragnn_v1.js"
echo "[demos_v1] HydraGNN done"

echo "[demos_v1] Recording ORBIT-2 DC 4K demo ..."
node "$SCRIPTS/record_orbit2_dc_v1.js"
echo "[demos_v1] ORBIT-2 done"

echo "[demos_v1] Recording GP-MoLFormer 4K demo ..."
node "$SCRIPTS/record_gpmolformer_v1.js"
echo "[demos_v1] GP-MoLFormer done"

# ── Copy deliverables to ~/transfer ───────────────────────────────────────────
echo "[demos_v1] Copying to $TRANSFER ..."
for f in hydragnn_demo_v1 orbit2_dc_demo_v1 gpmolformer_finetune_demo_v1; do
  if [ -f "$OUT_DIR/$f.mp4" ]; then
    cp -f "$OUT_DIR/$f.mp4" "$TRANSFER/"
    echo "[demos_v1]   -> $TRANSFER/$f.mp4"
  else
    echo "[demos_v1]   !! missing $OUT_DIR/$f.mp4"
  fi
done

echo ""
echo "[demos_v1] All 4K recordings complete at $(date)"
ls -lh "$TRANSFER"/*_v1.mp4 2>/dev/null || true
