#!/usr/bin/env bash
# AI4Science Studio — web GUI launcher.
#
# Runs the Streamlit app on THIS login node, bound to localhost only.
# Reach it from your laptop with an SSH port-forward:
#
#     ssh -L 8501:localhost:8501 <user>@<this-login-node>
#     # then open http://localhost:8501 in your browser
#
# Override the port with STUDIO_PORT=xxxx ./studio.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
APP_DIR="$SCRIPT_DIR/studio"
VENV="$APP_DIR/.venv"
PORT="${STUDIO_PORT:-8501}"

if [[ ! -d "$APP_DIR" ]]; then
  echo "error: expected app directory at $APP_DIR" >&2
  exit 1
fi

# Bootstrap a self-contained venv (streamlit + pyyaml only; no ML deps).
if [[ ! -x "$VENV/bin/streamlit" ]]; then
  echo "--- First run: creating venv at $VENV and installing deps ---"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$APP_DIR/requirements.txt"
fi

# Ensure browser->localhost traffic bypasses any corporate HTTP proxy that the
# cluster sets in the environment (otherwise the forwarded port is unreachable).
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"

HOSTNAME_NOW=$(hostname 2>/dev/null || echo "<login-node>")
echo ""
echo "=========================================================================="
echo " AI4Science Studio is starting on http://localhost:${PORT}"
echo ""
echo " If your browser is on another machine, open an SSH tunnel first:"
echo "     ssh -L ${PORT}:localhost:${PORT} \$USER@${HOSTNAME_NOW}"
echo " then browse to http://localhost:${PORT}"
echo "=========================================================================="
echo ""

exec "$VENV/bin/streamlit" run "$APP_DIR/app.py" \
  --server.address 127.0.0.1 \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
