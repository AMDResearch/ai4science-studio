#!/usr/bin/env bash
# run_generation.sh — Unconditional or scaffold-constrained molecule generation
#
# Run this INSIDE the container (docker exec -it gp-molformer bash)
# OR pass it directly: docker exec gp-molformer bash /workspace/run_generation.sh
#
# OPTIONAL EDITS:
#   SCAFFOLD     — SMILES fragment for constrained generation (unset = unconditional)
#   NUM_BATCHES  — number of batches of 1000 molecules (default: 1 = 1000 molecules)
#   OUTPUT_FILE  — output CSV path (default: /workspace/generated.csv)

set -euo pipefail

SCAFFOLD="${SCAFFOLD:-}"
NUM_BATCHES="${NUM_BATCHES:-1}"
OUTPUT_FILE="${OUTPUT_FILE:-/workspace/generated.csv}"

cd /workspace/gp-molformer

if [[ -n "$SCAFFOLD" ]]; then
    echo "=== Scaffold-Constrained Generation ==="
    echo "  Scaffold    : $SCAFFOLD"
    echo "  Output      : $OUTPUT_FILE"
    echo ""
    python scripts/conditional_generation.py "$SCAFFOLD" \
        | tee "$OUTPUT_FILE"
    echo ""
    echo "Valid molecules written to: $OUTPUT_FILE"
else
    echo "=== Unconditional Generation ==="
    echo "  Batches     : $NUM_BATCHES (${NUM_BATCHES}000 molecules)"
    echo "  Output      : $OUTPUT_FILE"
    echo ""
    python scripts/unconditional_generation.py \
        --num_batches "$NUM_BATCHES" \
        "$OUTPUT_FILE"
    echo ""
    echo "Generated SMILES written to: $OUTPUT_FILE"
fi

# Quick validity check
if command -v python &>/dev/null && python -c "import rdkit" &>/dev/null; then
    echo ""
    echo "Validity summary:"
    python - "$OUTPUT_FILE" <<'PYEOF'
import sys, csv
from rdkit import Chem
smiles = [row[0] for row in csv.reader(open(sys.argv[1])) if row]
valid = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
print(f"  Total   : {len(smiles)}")
print(f"  Valid   : {len(valid)} ({100*len(valid)/max(len(smiles),1):.1f}%)")
print(f"  Unique  : {len(set(valid))}")
PYEOF
fi
