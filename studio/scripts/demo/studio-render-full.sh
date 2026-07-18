#!/bin/bash
set -e
echo "NODE: $(hostname)  ulimit-u: $(ulimit -u)"

DEPS="/scratch/$USER/studio_chrome_deps_$$"
CHROME_LOCAL="/scratch/$USER/studio_chrome_$$"
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
cd /home/spannala/Projects/ai4science-studio/studio/demo
node /home/spannala/studio-render-test.js
