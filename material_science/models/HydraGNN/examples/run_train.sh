#!/usr/bin/env bash
# run_train.sh — Small-scale HydraGNN training on a single AMD Instinct GPU
#
# Run inside the container launched by docker_run.sh:
#   docker exec -it hydragnn bash /examples/run_train.sh
#
# NOTE: Published GFM 2024 pretraining used Frontier (OLCM) at very large scale
# with ADIOS preprocessed datasets. This script targets smaller-scale experiments
# (e.g. reproducing a single HPO trial) using the open datasets in the HydraGNN repo.
# See recipes/train/README.md for the full Frontier workflow.
#
# OPTIONAL EDITS:
#   HG_EXAMPLE     — which upstream example to run (default: multidataset_hpo)
#   HG_CONFIG      — path to a JSON config file (required if HG_EXAMPLE needs one)
#   HG_DATA_DIR    — staging directory for datasets (default: /data)
#   HG_OUTPUT_DIR  — checkpoints and logs (default: /workspace/checkpoints)

set -euo pipefail

HG_EXAMPLE="${HG_EXAMPLE:-multidataset_hpo}"
HG_CONFIG="${HG_CONFIG:-}"
HG_DATA_DIR="${HG_DATA_DIR:-/data}"
HG_OUTPUT_DIR="${HG_OUTPUT_DIR:-/workspace/checkpoints}"

mkdir -p "$HG_DATA_DIR" "$HG_OUTPUT_DIR"
cd /workspace/HydraGNN

echo "=== HydraGNN — Training ==="
echo "  Example    : $HG_EXAMPLE"
echo "  Data dir   : $HG_DATA_DIR"
echo "  Output dir : $HG_OUTPUT_DIR"
echo ""
echo "NOTE: Full GFM 2024 pretraining requires staged ADIOS data from Constellation"
echo "  and large-scale compute (see recipes/train/README.md)."
echo "  For local experiments, use one of the smaller datasets bundled in the repo."
echo ""

if [[ -z "${HG_CONFIG}" ]]; then
    echo "ERROR: HG_CONFIG is not set."
    echo "  Point it at a JSON config file for your chosen example, e.g.:"
    echo "  HG_CONFIG=/workspace/HydraGNN/examples/${HG_EXAMPLE}/config.json \\"
    echo "  bash /examples/run_train.sh"
    exit 1
fi

echo "Using config: $HG_CONFIG"
echo ""

python -u "examples/${HG_EXAMPLE}/train.py" \
    --config "$HG_CONFIG" \
    --log_dir "$HG_OUTPUT_DIR"
