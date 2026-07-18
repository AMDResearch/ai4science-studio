#!/bin/bash
DEPS="/scratch/$USER/shot_deps_$$"; mkdir -p "$DEPS"; cd /tmp
apt-get download libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libasound2t64 \
  libxcomposite1 libxdamage1 libxrandr2 libatspi2.0-0t64 libgbm1 libxkbcommon0 libxi6 2>&1 | grep -c Get || true
for d in /tmp/*.deb; do [ -f "$d" ] && dpkg-deb -x "$d" "$DEPS/"; done
export LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu:$DEPS/usr/lib:${LD_LIBRARY_PATH:-}"
CHROME_LOCAL="/scratch/$USER/shot_chrome_$$"; mkdir -p "$CHROME_LOCAL"
cp -r ~/.cache/ms-playwright/chromium-1228 "$CHROME_LOCAL/"
export PLAYWRIGHT_BROWSERS_PATH="$CHROME_LOCAL"
cd /home/spannala/Projects/ai4science-studio/studio/demo
RID='e1a1c76a-e9e9-46f5-a9e5-77c2044cfe1b' node /home/spannala/orbit2_analyze_shot.js
rm -rf "$DEPS" "$CHROME_LOCAL"
