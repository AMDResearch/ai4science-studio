#!/bin/bash
# Screenshot the Analyze step of one studio run (RID) on a compute node.
# Usage: RID=<studio run id> bash studio/scripts/demo/shot_run.sh
set -u
: "${RID:?set RID to the studio run id to capture}"
STUDIO="${STUDIO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEPS="${AI4S_LOCAL_SCRATCH:-${TMPDIR:-/tmp}/$USER}/shot_deps_$$"; mkdir -p "$DEPS"; cd /tmp
apt-get download libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libasound2t64 \
  libxcomposite1 libxdamage1 libxrandr2 libatspi2.0-0t64 libgbm1 libxkbcommon0 libxi6 2>&1 | grep -c Get || true
for d in /tmp/*.deb; do [ -f "$d" ] && dpkg-deb -x "$d" "$DEPS/"; done
export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu:$DEPS/usr/lib:${LD_LIBRARY_PATH:-}"
CHROME_LOCAL="${AI4S_LOCAL_SCRATCH:-${TMPDIR:-/tmp}/$USER}/shot_chrome_$$"; mkdir -p "$CHROME_LOCAL"
cp -r ~/.cache/ms-playwright/chromium-1228 "$CHROME_LOCAL/"
export PLAYWRIGHT_BROWSERS_PATH="$CHROME_LOCAL"
cd "$STUDIO/demo"
RID="$RID" node "$STUDIO/scripts/demo/orbit2_analyze_shot.js"
rm -rf "$DEPS" "$CHROME_LOCAL"
