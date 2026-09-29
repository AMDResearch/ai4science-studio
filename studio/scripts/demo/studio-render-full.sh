#!/bin/bash
set -e
STUDIO="${STUDIO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
echo "NODE: $(hostname)  ulimit-u: $(ulimit -u)"

DEPS="${AI4S_LOCAL_SCRATCH:-${TMPDIR:-/tmp}/$USER}/studio_chrome_deps_$$"
CHROME_LOCAL="${AI4S_LOCAL_SCRATCH:-${TMPDIR:-/tmp}/$USER}/studio_chrome_$$"
mkdir -p "$DEPS" "$CHROME_LOCAL"

echo "[test] Staging GTK/X libs..."
cd /tmp
apt-get download \
  libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libasound2t64 \
  libxcomposite1 libxdamage1 libxrandr2 libatspi2.0-0t64 \
  libgbm1 libxkbcommon0 libxi6 2>&1 | grep -v "^Get:\|^Fetched\|^Reading\|^W:" || true
for d in /tmp/*.deb; do [ -f "$d" ] && dpkg-deb -x "$d" "$DEPS/"; done
export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu:$DEPS/usr/lib:${LD_LIBRARY_PATH:-}"

echo "[test] Copying Chrome to scratch (bypass NFS mmap)..."
cp -r ~/.cache/ms-playwright/chromium-1228 "$CHROME_LOCAL/"
export PLAYWRIGHT_BROWSERS_PATH="$CHROME_LOCAL"

echo "[test] Backend health:"
curl -sf http://localhost:8275/api/health && echo ""

echo "[test] Running render test..."
cd "$STUDIO/demo"
node "$STUDIO/scripts/demo/studio-render-test.js"
