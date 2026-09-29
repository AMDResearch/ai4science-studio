#!/bin/bash
# Set up the demo recorder dependencies.
# Matches the DigitalTwin demo/setup_demo.sh pattern.
#
# Usage:
#   cd <repo>/studio/demo && bash setup_demo.sh

set -e
cd "$(dirname "$0")"
echo "[setup] npm install playwright..."
npm install 2>&1 | tail -3

echo "[setup] Installing Playwright chromium + ffmpeg..."
npx playwright install chromium 2>&1 | tail -5
npx playwright install ffmpeg 2>&1 | tail -3 || true

# Offline fallback: copy an existing node_modules (e.g. from another checkout)
# when npm cannot reach the registry.
if [ ! -d node_modules ] && [ -n "${DEMO_NODE_MODULES_SRC:-}" ] && [ -d "$DEMO_NODE_MODULES_SRC" ]; then
  echo "[setup] Copying node_modules from $DEMO_NODE_MODULES_SRC..."
  cp -r "$DEMO_NODE_MODULES_SRC" node_modules
fi

echo "DONE_DEMO_SETUP"
