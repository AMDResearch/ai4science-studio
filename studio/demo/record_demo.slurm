#!/bin/bash
# AMD AI4Science Studio — Slurm batch demo recorder
# Adapted from a tested DigitalTwin demo recorder (2026-07-16).
#
# Usage:
#   sbatch studio/demo/record_demo.slurm
#   # or with explicit partition override:
#   sbatch --partition=<partition> --account=<account> studio/demo/record_demo.slurm
#
# Output: <repo>/studio/demo/demo-output/ai4science_demo.{webm,mp4}

#SBATCH --job-name=studio_demo
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

set -e
echo "[demo] Job $SLURM_JOB_ID on $(hostname) at $(date)"

# Studio services run on another host (e.g. the login node). Compute nodes may
# not resolve its hostname, so STUDIO_HOST is usually its internal IP address.
LOGIN_HOST="${STUDIO_HOST:?set STUDIO_HOST to the host (IP) running the studio services}"
export STUDIO_URL="http://${LOGIN_HOST}:5173"

STUDIO="${STUDIO_DIR:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)/studio}"
DEPS="${AI4S_LOCAL_SCRATCH:-${TMPDIR:-/tmp}/$USER}/studio_chrome_deps_$$"
CHROME_LOCAL="${AI4S_LOCAL_SCRATCH:-${TMPDIR:-/tmp}/$USER}/studio_chrome_$$"
mkdir -p "$DEPS" "$CHROME_LOCAL"

# ── Stage missing GTK/X libs (Ubuntu 24.04 t64 naming, no root needed) ──────
echo "[demo] Staging missing shared libs..."
cd /tmp
apt-get download \
  libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libasound2t64 \
  libxcomposite1 libxdamage1 libxrandr2 libatspi2.0-0t64 \
  libgbm1 libxkbcommon0 libxi6 2>&1 | grep -v "^Get:\|^Fetched\|^Reading\|^W:" || true
for d in /tmp/*.deb; do [ -f "$d" ] && dpkg-deb -x "$d" "$DEPS/"; done
export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu:$DEPS/usr/lib:${LD_LIBRARY_PATH:-}"
echo "[demo] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# ── Copy Chrome + Playwright ffmpeg to /scratch (bypass NFS mmap) ───────────
echo "[demo] Copying Chrome to scratch..."
cp -r ~/.cache/ms-playwright/chromium-1228 "$CHROME_LOCAL/"
cp -r ~/.cache/ms-playwright/ffmpeg-1011  "$CHROME_LOCAL/"   # must copy BOTH
export PLAYWRIGHT_BROWSERS_PATH="$CHROME_LOCAL"
echo "[demo] PLAYWRIGHT_BROWSERS_PATH=$PLAYWRIGHT_BROWSERS_PATH"

# ── Check services on login node (already started before sbatch) ─────────────
echo "[demo] Checking studio services on $LOGIN_HOST..."
for retry in 1 2 3; do
  curl -sf "http://${LOGIN_HOST}:8099/api/health" && echo " — backend healthy" && break || sleep 3
done
curl -sf "http://${LOGIN_HOST}:8099/api/health" > /dev/null || {
  echo "[demo] Backend not reachable on ${LOGIN_HOST}:8099 — start the studio services first"; exit 1
}
for retry in 1 2 3; do
  curl -sf "http://${LOGIN_HOST}:5173/" > /dev/null && echo "[demo] Frontend healthy" && break || sleep 5
done
curl -sf "http://${LOGIN_HOST}:5173/" > /dev/null || {
  echo "[demo] Frontend not reachable on ${LOGIN_HOST}:5173 — start the studio services first"; exit 1
}
BACK_PID=""
FRONT_PID=""

# ── Activate env for FFMPEG env var (imageio-ffmpeg path) ────────────────────
# Try the studio venv's ffmpeg first, then Software venv
FFMPEG_CANDIDATES=(
  "$STUDIO/backend/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)
# An explicit $FFMPEG wins; `ffmpeg` on PATH is the last resort.
[ -n "${FFMPEG:-}" ] && FFMPEG_CANDIDATES=("$FFMPEG" "${FFMPEG_CANDIDATES[@]}")
FFMPEG_CANDIDATES+=("$(command -v ffmpeg || true)")
for f in "${FFMPEG_CANDIDATES[@]}"; do
  if [ -f "$f" ]; then export FFMPEG="$f"; echo "[demo] FFMPEG=$FFMPEG"; break; fi
done

# ── Record ────────────────────────────────────────────────────────────────────
echo "[demo] Starting recorder..."
cd "$STUDIO/demo"
node record-demo.js 2>&1
EXIT=$?
echo "[demo] Recorder exited with code $EXIT"

# ── Cleanup ───────────────────────────────────────────────────────────────────
[ -n "$FRONT_PID" ] && kill $FRONT_PID 2>/dev/null || true
[ -n "$BACK_PID" ] && kill $BACK_PID 2>/dev/null || true
rm -rf "$DEPS" "$CHROME_LOCAL"
echo "[demo] Cleanup done at $(date)"
echo "[demo] Output files:"
ls -lh "$STUDIO/demo/demo-output/" 2>/dev/null || true
exit $EXIT
