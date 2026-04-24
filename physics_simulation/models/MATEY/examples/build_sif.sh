#!/usr/bin/env bash
# Build an Apptainer SIF for MATEY and install dependencies into a writable overlay.
#
# What this does
# --------------
# 1. Pulls the ROCm PyTorch Docker image and converts it to a SIF file.
# 2. Creates a writable ext3 overlay image (default 8 GB).
# 3. Installs MATEY (cloned from GitHub) into the overlay via `apptainer exec`.
#
# The resulting pair (SIF + overlay) is then referenced in sbatch_train_amd.sh.
#
# Usage
# -----
#   ./build_sif.sh                  # uses defaults below
#   MATEY_SIF=~/matey.sif OVERLAY_SIZE_MB=16384 ./build_sif.sh
#
# Environment variables
# ---------------------
#   MATEY_SIF        path where the SIF file will be written  (default: ./matey.sif)
#   MATEY_OVERLAY    path where the overlay image will be written (default: ./matey_overlay.img)
#   OVERLAY_SIZE_MB  size of the writable overlay in MB  (default: 8192 = 8 GB)
#   ROCM_IMAGE       Docker image to convert  (default: rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0)
#   MATEY_REPO       upstream Git repo to clone  (default: https://github.com/ORNL/MATEY.git)
#   MATEY_SRC        local clone path  (default: ./MATEY)

set -euo pipefail

MATEY_SIF="${MATEY_SIF:-$(pwd)/matey.sif}"
MATEY_OVERLAY="${MATEY_OVERLAY:-$(pwd)/matey_overlay.img}"
OVERLAY_SIZE_MB="${OVERLAY_SIZE_MB:-8192}"
ROCM_IMAGE="${ROCM_IMAGE:-rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.6.0}"
MATEY_REPO="${MATEY_REPO:-https://github.com/ORNL/MATEY.git}"
MATEY_SRC="${MATEY_SRC:-$(pwd)/MATEY}"

echo "=== MATEY SIF build ==="
echo "  Docker image   : ${ROCM_IMAGE}"
echo "  SIF output     : ${MATEY_SIF}"
echo "  Overlay output : ${MATEY_OVERLAY}"
echo "  Overlay size   : ${OVERLAY_SIZE_MB} MB"
echo "  MATEY source   : ${MATEY_SRC}"
echo ""

# ---------------------------------------------------------------------------
# 1. Clone MATEY if not already present
# ---------------------------------------------------------------------------
if [[ ! -d "${MATEY_SRC}" ]]; then
    echo "Cloning MATEY …"
    git clone "${MATEY_REPO}" "${MATEY_SRC}"
else
    echo "MATEY source already present at ${MATEY_SRC} — skipping clone."
fi

# ---------------------------------------------------------------------------
# 2. Pull Docker image → SIF
# ---------------------------------------------------------------------------
if [[ -f "${MATEY_SIF}" ]]; then
    echo "SIF already exists at ${MATEY_SIF} — skipping pull."
    echo "  Delete it and re-run to rebuild from scratch."
else
    echo "Pulling Docker image and converting to SIF (this may take 10–20 min) …"
    apptainer pull "${MATEY_SIF}" "docker://${ROCM_IMAGE}"
    echo "SIF written to ${MATEY_SIF}"
fi

# ---------------------------------------------------------------------------
# 3. Create writable ext3 overlay
# ---------------------------------------------------------------------------
if [[ -f "${MATEY_OVERLAY}" ]]; then
    echo "Overlay already exists at ${MATEY_OVERLAY} — skipping creation."
    echo "  Delete it and re-run to recreate the overlay."
else
    echo "Creating ${OVERLAY_SIZE_MB} MB writable overlay …"
    # Create a blank file and format as ext3
    dd if=/dev/zero of="${MATEY_OVERLAY}" bs=1M count="${OVERLAY_SIZE_MB}" status=progress
    mkfs.ext3 -F "${MATEY_OVERLAY}"
    echo "Overlay written to ${MATEY_OVERLAY}"
fi

# ---------------------------------------------------------------------------
# 4. Install MATEY into the overlay
# ---------------------------------------------------------------------------
echo "Installing MATEY into overlay …"
apptainer exec \
    --overlay "${MATEY_OVERLAY}" \
    --bind "${MATEY_SRC}:/matey-src" \
    "${MATEY_SIF}" \
    bash -c "
        set -e
        pip install --quiet -e /matey-src
        echo 'MATEY installed:'
        pip show matey 2>/dev/null || python /matey-src/basic_usage.py --help 2>&1 | head -5
    "

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== Build complete ==="
echo ""
echo "Set these variables before submitting the SLURM job:"
echo "  export MATEY_SIF=${MATEY_SIF}"
echo "  export MATEY_OVERLAY=${MATEY_OVERLAY}"
echo ""
echo "Then submit:"
echo "  sbatch sbatch_train_amd.sh"
echo ""
echo "To update MATEY after a git pull (without rebuilding the SIF):"
echo "  ${0} --update-only"
echo "  # or manually:"
echo "  apptainer exec --overlay ${MATEY_OVERLAY} ${MATEY_SIF} pip install -e /matey-src"
