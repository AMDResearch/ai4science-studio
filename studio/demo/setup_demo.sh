#!/bin/bash
# Set up the demo recorder dependencies.
# Matches the DigitalTwin demo/setup_demo.sh pattern.
#
# Usage:
#   cd ~/Projects/ai4science-studio/studio/demo && bash setup_demo.sh

set -e
cd "$(dirname "$0")"
echo "[setup] npm install playwright..."
npm install 2>&1 | tail -3

echo "[setup] Installing Playwright chromium + ffmpeg..."
npx playwright install chromium 2>&1 | tail -5
npx playwright install ffmpeg 2>&1 | tail -3 || true

echo "[setup] Checking node_modules from DigitalTwin (fallback)..."
if [ ! -d node_modules ]; then
  DT="$HOME/Projects/Fusion/DigitalTwin/demo/node_modules"
  if [ -d "$DT" ]; then
    echo "[setup] Copying node_modules from DigitalTwin..."
    cp -r "$DT" .
  fi
fi

echo "DONE_DEMO_SETUP"
