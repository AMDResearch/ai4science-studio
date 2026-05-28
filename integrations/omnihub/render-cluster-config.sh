#!/usr/bin/env bash
# Validate ai4science .cluster-config.yaml omnihub section against OmniHub vultr.yaml.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AI4S_REPO_ROOT="${AI4S_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OMNIHUB_DIR="${OMNIHUB_DIR:-$HOME/git/omnihub}"

CLUSTER_CFG="${AI4S_CLUSTER_CONFIG:-$AI4S_REPO_ROOT/.cluster-config.yaml}"
VULTR_YAML="$OMNIHUB_DIR/config/vultr.yaml"

_yaml_get() {
  local key=$1 file=$2
  grep -E "^[[:space:]]*${key}:" "$file" 2>/dev/null | head -1 | sed -E 's/^[^:]*:[[:space:]]*"?([^"]*)"?/\1/' | tr -d '"'
}

echo "=== OmniHub cluster config bridge ==="
echo "  ai4science config: $CLUSTER_CFG"
echo "  omnihub vultr.yaml: $VULTR_YAML"
echo ""

if [[ ! -f "$CLUSTER_CFG" ]]; then
  echo "WARN: no .cluster-config.yaml — copy from .cluster-config.example.yaml" >&2
fi

if [[ ! -f "$VULTR_YAML" ]]; then
  echo "ERROR: missing $VULTR_YAML" >&2
  exit 2
fi

SHARED_OMNI=$(_yaml_get shared-dir "$VULTR_YAML")
RESULTS_OMNI=$(_yaml_get results-dir "$VULTR_YAML")

echo "OmniHub vultr.yaml:"
echo "  shared-dir:  $SHARED_OMNI"
echo "  results-dir: $RESULTS_OMNI"
echo ""

if [[ -f "$CLUSTER_CFG" ]]; then
  REPO_DIR=$(_yaml_get repo_dir "$CLUSTER_CFG" 2>/dev/null || true)
  # nested under omnihub: — grep with context
  OMNI_REPO=$(awk '/^omnihub:/{f=1} f && /repo_dir:/{print; exit}' "$CLUSTER_CFG" | sed -E 's/.*:[[:space:]]*"?([^"]*)"?/\1/')
  OMNI_SHARED=$(awk '/^omnihub:/{f=1} f && /shared_dir:/{print; exit}' "$CLUSTER_CFG" | sed -E 's/.*:[[:space:]]*"?([^"]*)"?/\1/')

  echo "ai4science omnihub section:"
  echo "  repo_dir:    ${OMNI_REPO:-<unset>}"
  echo "  shared_dir:  ${OMNI_SHARED:-<unset>}"
  echo ""

  if [[ -n "${OMNI_SHARED:-}" && "$OMNI_SHARED" != "$SHARED_OMNI" ]]; then
    echo "WARN: omnihub.shared_dir ($OMNI_SHARED) != vultr shared-dir ($SHARED_OMNI)" >&2
  fi
fi

for tool_path in "$SHARED_OMNI/tools/omnistat/venv/bin/omnistat-usermode" \
                 "$SHARED_OMNI/tools/omnistat-rocprofiler/venv/bin/omnistat-usermode"; do
  if [[ -x "$tool_path" ]]; then
    echo "OK: $tool_path"
  else
    echo "MISSING: $tool_path" >&2
  fi
done

echo ""
echo "Run ./integrations/omnihub/omnistat-parity-check.sh for detailed Omnistat comparison."
