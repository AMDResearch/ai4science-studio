#!/usr/bin/env bash
# HydraGNN studio inference: predict formation energy on a real held-out Alexandria
# material using OUR locally-trained PNA model, reporting prediction vs DFT reference.
#
# Env (set by the studio job wrapper):
#   HG_SIF              Apptainer SIF image
#   HG_OVERLAY          HydraGNN ext3 overlay (has torch/ase/adios2 deps)
#   HG_INFER_REPO       Cloned HydraGNN repo (Predictive_GFM_2024 branch)
#   HG_MODEL_PATH       Trained model checkpoint (.pk with model_state_dict + config)
#   HG_STUDIO_INFER     studio_infer.py path
#   STUDIO_RESULT_OUT   where to write result JSON
#   STRUCT_INDEX        which held-out structure to predict (default 4)
set -euo pipefail

HG_SIF="${HG_SIF:?}"
HG_OVERLAY="${HG_OVERLAY:?}"
HG_INFER_REPO="${HG_INFER_REPO:?}"
HG_MODEL_PATH="${HG_MODEL_PATH:?}"
HG_STUDIO_INFER="${HG_STUDIO_INFER:?}"

echo "=== HydraGNN energy prediction (our trained model) ==="
echo "  Model      : $HG_MODEL_PATH"
echo "  Structure  : held-out Alexandria index ${STRUCT_INDEX:-4}"
echo ""

# Clear leaked SLURM/PMIx/OpenMPI launcher env so mpi4py inits in singleton mode
# inside the plain apptainer exec (same fix used by the other model recipes).
_MPI_UNSET=()
while IFS= read -r _v; do _MPI_UNSET+=(-u "$_v"); done < <(
  env | grep -oE '^(SLURM_|PMIX_|PMI_|OMPI_)[A-Za-z0-9_]+' || true
)

env "${_MPI_UNSET[@]}" apptainer exec --rocm \
    --overlay "${HG_OVERLAY}:ro" \
    --bind /shared/aaji/models/HydraGNN/weights:/shared/aaji/models/HydraGNN/weights:ro \
    --bind /shared/spannala/models/HydraGNN:/shared/spannala/models/HydraGNN \
    --env HG_INFER_REPO="$HG_INFER_REPO" \
    --env MODEL_PATH="$HG_MODEL_PATH" \
    --env HG_MODEL_VARIANT="${HG_MODEL_VARIANT:-}" \
    --env STRUCT_INDEX="${STRUCT_INDEX:-4}" \
    --env STUDIO_RESULT_OUT="${STUDIO_RESULT_OUT:-}" \
    --env PYTHONPATH="${HG_INFER_REPO}:/opt/hydragnn-pkgs" \
    --env LD_LIBRARY_PATH=/opt/hydragnn-pkgs/adios2:/opt/venv/lib/python3.12/site-packages/torch/lib \
    "$HG_SIF" \
    bash -c "source /opt/venv/bin/activate && python3 $HG_STUDIO_INFER"
