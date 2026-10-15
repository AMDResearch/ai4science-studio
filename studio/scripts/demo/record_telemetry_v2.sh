#!/bin/bash
# Record the HydraGNN 8-GPU TRAINING + live Omnistat telemetry 4K demo
# (3840x2160, 16:9). Same environment recipe as record_demos_v1.sh (GTK/X libs +
# Chromium to scratch + imageio ffmpeg), targeting the fixed live ports.
#
# Run ON the compute node where the studio services are already running:
#   srun --jobid=<hold-jobid> --overlap bash \
#     <repo>/studio/scripts/demo/record_telemetry_v2.sh
#
# Output: ~/transfer/hydragnn_training_4k_demo.mp4
set -e
echo "[tele_v2] Host $(hostname) at $(date)"

STUDIO="${STUDIO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRIPTS="$STUDIO/scripts/demo"
export FRONT_URL="${FRONT_URL:-http://127.0.0.1:5376}"
export BACK_URL="${BACK_URL:-http://127.0.0.1:8376}"
export OUT_DIR="${OUT_DIR:-$STUDIO/demo/demo-output}"
TRANSFER="${TRANSFER_DIR:-$HOME/transfer}"
mkdir -p "$OUT_DIR" "$TRANSFER"

DEPS="${AI4S_LOCAL_SCRATCH:-${TMPDIR:-/tmp}/$USER}/rtele_v2_deps_$$"
CHROME_LOCAL="${AI4S_LOCAL_SCRATCH:-${TMPDIR:-/tmp}/$USER}/rtele_v2_chrome_$$"
mkdir -p "$DEPS" "$CHROME_LOCAL"
cleanup() { rm -rf "$DEPS" "$CHROME_LOCAL"; }
trap cleanup EXIT

echo "[tele_v2] Staging GTK/X libs..."
cd /tmp
apt-get download libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libasound2t64 \
  libxcomposite1 libxdamage1 libxrandr2 libatspi2.0-0t64 libgbm1 libxkbcommon0 libxi6 \
  2>&1 | grep -vE "^Get:|^Fetched|^Reading|^W:" || true
for d in /tmp/*.deb; do [ -f "$d" ] && dpkg-deb -x "$d" "$DEPS/" 2>/dev/null; done
export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu:$DEPS/usr/lib:${LD_LIBRARY_PATH:-}"
cp -r ~/.cache/ms-playwright/chromium-1228 "$CHROME_LOCAL/"
export PLAYWRIGHT_BROWSERS_PATH="$CHROME_LOCAL"

FFMPEG_CANDIDATES=(
  "$STUDIO/backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
# An explicit $FFMPEG wins; `ffmpeg` on PATH is the last resort.
[ -n "${FFMPEG:-}" ] && FFMPEG_CANDIDATES=("$FFMPEG" "${FFMPEG_CANDIDATES[@]}")
FFMPEG_CANDIDATES+=("$(command -v ffmpeg || true)")
for f in "${FFMPEG_CANDIDATES[@]}"; do
  if [ -f "$f" ]; then export FFMPEG="$f"; echo "[tele_v2] FFMPEG=$FFMPEG"; break; fi
done

curl -sf "$BACK_URL/api/health" >/dev/null 2>&1 || { echo "[tele_v2] Backend not reachable at $BACK_URL"; exit 1; }
curl -sf "$FRONT_URL/" >/dev/null 2>&1 || { echo "[tele_v2] Frontend not reachable at $FRONT_URL"; exit 1; }
echo "[tele_v2] services healthy"

cd "$STUDIO/demo"
echo "[tele_v2] Recording HydraGNN telemetry 4K demo ..."
node "$SCRIPTS/record_hydragnn_telemetry_v2.js"

if [ -f "$OUT_DIR/hydragnn_telemetry_v2.mp4" ]; then
  cp -f "$OUT_DIR/hydragnn_telemetry_v2.mp4" "$TRANSFER/hydragnn_training_4k_demo.mp4"
  echo "[tele_v2] -> $TRANSFER/hydragnn_training_4k_demo.mp4"
else
  echo "[tele_v2] !! missing $OUT_DIR/hydragnn_telemetry_v2.mp4"; exit 1
fi

echo "[tele_v2] Done at $(date)"
ls -lh "$TRANSFER/hydragnn_training_4k_demo.mp4" 2>/dev/null || true
