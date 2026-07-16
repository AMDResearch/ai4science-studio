#!/usr/bin/env bash
# AI4Science Studio — React/FastAPI web GUI launcher.
#
# Runs the FastAPI backend (uvicorn) on THIS login node, bound to localhost
# only, and serves the pre-built React SPA from studio/web/dist. Reach it from
# your laptop with an SSH port-forward:
#
#     ssh -L 8600:localhost:8600 <user>@<this-login-node>
#     # then open http://localhost:8600 in your browser
#
# Override the port with STUDIO_WEB_PORT=xxxx ./studio_web.sh
#
# ---------------------------------------------------------------------------
# Approach A (RECOMMENDED): build locally, ship the dist/, run here
# ---------------------------------------------------------------------------
# The login node has NO npm. The React build is a pure static-bundle step that
# needs no cluster access, so do it on your LOCAL laptop and copy the result up.
# The app must RUN on the login node because that is where SLURM (sbatch/squeue)
# lives — so we ship the built dist/ here and let uvicorn serve it.
#
#   # 1) On your LOCAL laptop (has npm):
#   cd studio/web && npm install && npm run build      # -> studio/web/dist/
#
#   # 2) Copy the build up to the login node:
#   scp -r studio/web/dist/ <user>@<login-node>:<repo-path>/studio/web/dist/
#
#   # 3) On the login node (here):
#   ./studio_web.sh            # uvicorn serves dist/ at 127.0.0.1:8600
#
#   # 4) From your laptop, tunnel in:
#   ssh -L 8600:localhost:8600 <user>@<login-node>   # open http://localhost:8600
#
# Approach B (for active UI development): run the Vite dev server on your laptop
# with its /api proxy pointed through the SSH tunnel to this backend. See
# studio/web/README.md.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STUDIO_DIR="$SCRIPT_DIR/studio"
API_DIR="$STUDIO_DIR/api"
VENV="$API_DIR/.venv"
DIST="$STUDIO_DIR/web/dist"
PORT="${STUDIO_WEB_PORT:-8600}"

if [[ ! -d "$API_DIR" ]]; then
  echo "error: expected API directory at $API_DIR" >&2
  exit 1
fi

# Bootstrap a self-contained venv (fastapi + uvicorn + pyyaml; no ML deps).
if [[ ! -x "$VENV/bin/uvicorn" ]]; then
  echo "--- First run: creating venv at $VENV and installing deps ---"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$API_DIR/requirements.txt"
fi

# Warn (don't fail) if the React build hasn't been shipped up yet — the backend
# still serves the JSON API, and Approach B (Vite dev server) works without it.
if [[ ! -d "$DIST" ]]; then
  echo ""
  echo "!! studio/web/dist not found — the React SPA has not been built/copied."
  echo "   Recommended (Approach A): build on your LOCAL laptop, then scp up:"
  echo "       cd studio/web && npm install && npm run build"
  echo "       scp -r studio/web/dist/ <user>@<login-node>:<repo-path>/studio/web/dist/"
  echo "   The JSON API will still serve at http://localhost:${PORT}/api"
  echo "   (or use Approach B: the Vite dev server — see studio/web/README.md)."
  echo ""
fi

# Ensure browser->localhost traffic bypasses any corporate HTTP proxy that the
# cluster sets in the environment (otherwise the forwarded port is unreachable).
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"

HOSTNAME_NOW=$(hostname 2>/dev/null || echo "<login-node>")
echo ""
echo "=========================================================================="
echo " AI4Science Studio (React/FastAPI) starting on http://localhost:${PORT}"
echo ""
echo " If your browser is on another machine, open an SSH tunnel first:"
echo "     ssh -L ${PORT}:localhost:${PORT} \$USER@${HOSTNAME_NOW}"
echo " then browse to http://localhost:${PORT}"
echo "=========================================================================="
echo ""

# Run uvicorn with studio/ on sys.path so `api.main` can import `core`.
cd "$STUDIO_DIR"
exec "$VENV/bin/uvicorn" api.main:app \
  --host 127.0.0.1 \
  --port "$PORT"
