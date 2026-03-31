#!/usr/bin/env bash
# run_inference.sh — Generate novel crystal structures with MatterGen
#
# Run inside the container: docker exec -it mattergen bash /workspace/run_inference.sh
#
# OPTIONAL EDITS:
#   PRETRAINED_NAME — checkpoint to use (default: mattergen_base)
#   BATCH_SIZE      — structures per batch (default: 16)
#   NUM_BATCHES     — number of batches (default: 1 → 16 structures)
#   OUTPUT_DIR      — output directory (default: /workspace/results)
#
# For conditioned generation, set PROPERTIES and GUIDANCE_FACTOR (see below).

set -euo pipefail

# --- Configuration ---
PRETRAINED_NAME="${PRETRAINED_NAME:-mattergen_base}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_BATCHES="${NUM_BATCHES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/results}"
# For property-conditioned generation (leave empty for unconditional):
#   PROPERTIES="{'dft_mag_density': 0.15}"
#   PRETRAINED_NAME="dft_mag_density"
PROPERTIES="${PROPERTIES:-}"
GUIDANCE_FACTOR="${GUIDANCE_FACTOR:-2.0}"

cd /workspace/mattergen
mkdir -p "$OUTPUT_DIR"

echo "=== MatterGen Generation ==="
echo "  Checkpoint  : $PRETRAINED_NAME"
echo "  Batch size  : $BATCH_SIZE"
echo "  Num batches : $NUM_BATCHES  (total: $((BATCH_SIZE * NUM_BATCHES)) structures)"
echo "  Output      : $OUTPUT_DIR"
if [[ -n "$PROPERTIES" ]]; then
    echo "  Properties  : $PROPERTIES"
    echo "  Guidance    : $GUIDANCE_FACTOR"
fi
echo ""

if [[ -n "$PROPERTIES" ]]; then
    mattergen-generate "$OUTPUT_DIR" \
        --pretrained-name="$PRETRAINED_NAME" \
        --batch_size="$BATCH_SIZE" \
        --num_batches="$NUM_BATCHES" \
        --properties_to_condition_on="$PROPERTIES" \
        --diffusion_guidance_factor="$GUIDANCE_FACTOR"
else
    mattergen-generate "$OUTPUT_DIR" \
        --pretrained-name="$PRETRAINED_NAME" \
        --batch_size="$BATCH_SIZE" \
        --num_batches="$NUM_BATCHES"
fi

echo ""
echo "Generated structures written to: $OUTPUT_DIR"
echo ""
echo "To evaluate with MatterSim relaxation:"
echo "  mattergen-evaluate \\"
echo "      --structures_path=$OUTPUT_DIR \\"
echo "      --relax=True \\"
echo "      --save_as=$OUTPUT_DIR/metrics.json"
echo ""
echo "--- Conditioned generation examples ---"
echo "Magnetic density:"
echo "  PRETRAINED_NAME=dft_mag_density PROPERTIES=\"{'dft_mag_density': 0.15}\" bash run_inference.sh"
echo "Composition + stability (Li-O, hull distance ≤ 0.05 eV/atom):"
echo "  PRETRAINED_NAME=chemical_system_energy_above_hull \\"
echo "  PROPERTIES=\"{'energy_above_hull': 0.05, 'chemical_system': 'Li-O'}\" bash run_inference.sh"
