#!/usr/bin/env bash
# GP-MoLFormer pair-tuning on AMD Instinct via SLURM + Apptainer.
#
# RESEARCH/ENGINEERING USE ONLY. Pair-tuning steers the molecule generator
# toward a target property. Outputs are for research only and must not be
# used for clinical or patient-treatment decisions.
#
# Pair-tuning trains N soft-prompt tokens prepended to GP-MoLFormer to bias
# generation toward molecules with higher QED / logP / DRD2-activity scores.
# Only the prompt tokens are trained; backbone weights remain frozen.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   export GPMOL_SIF=/path/to/pytorch_rocm7.2.2.sif
#   sbatch sbatch_pairtune_amd.sh           # QED property, 10 epochs
#   PAIRTUNE_PROP=logp sbatch ...           # logP property
#   PAIRTUNE_PROP=drd2 sbatch ...           # DRD2 binding activity
#
# ── Key environment variables ─────────────────────────────────────────────────
#   GPMOL_SIF           Path to Apptainer SIF image (required)
#   GPMOL_WORK_DIR      Host directory for repo clone, weights, output
#                       (default: ${AI4S_SHARED_DIR}/models/GP-MoLFormer)
#   PAIRTUNE_PROP       Property to optimize: qed | logp | drd2 (default: qed)
#   PAIRTUNE_EPOCHS     Number of training epochs (default: 10)
#   PAIRTUNE_BATCH      Batch size (default: 32)
#   PAIRTUNE_EVAL_FREQ  Evaluate every N epochs (default: 5)
#   STUDIO_RESULT_OUT   If set, writes a JSON result file for Studio harvest
#
#SBATCH --job-name=gpmol-pairtune
#SBATCH --partition=YOUR_PARTITION_HERE
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=gpmolformer-pairtune-%j.out
#SBATCH --error=gpmolformer-pairtune-%j.out

set -euo pipefail

if [[ -n "${STUDIO_EXAMPLES_DIR:-}" ]]; then
  SCRIPT_DIR=$(cd "$STUDIO_EXAMPLES_DIR" && pwd)
else
  SCRIPT_DIR=$(cd "$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null \
    | awk '/Command=/{print $1}' \
    | sed 's|Command=||' \
    | xargs -r dirname 2>/dev/null)" && pwd 2>/dev/null) \
    || SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

GPMOL_BASE="${AI4S_SHARED_DIR:-/your/shared/dir}/models/GP-MoLFormer"
GPMOL_SIF="${GPMOL_SIF:-${AI4S_SHARED_DIR:-/your/shared/dir}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
GPMOL_WORK_DIR="${GPMOL_WORK_DIR:-$GPMOL_BASE}"
PAIRTUNE_PROP="${PAIRTUNE_PROP:-qed}"
PAIRTUNE_EPOCHS="${PAIRTUNE_EPOCHS:-10}"
PAIRTUNE_BATCH="${PAIRTUNE_BATCH:-32}"
PAIRTUNE_EVAL_FREQ="${PAIRTUNE_EVAL_FREQ:-5}"

GPMOL_PKGDIR="${TMPDIR:-/tmp}/gpmol-pkgs-${SLURM_JOB_ID:-$$}"
mkdir -p "$GPMOL_PKGDIR"

echo "=== GP-MoLFormer pair-tuning ==="
echo "  SIF          : $GPMOL_SIF"
echo "  Work dir     : $GPMOL_WORK_DIR"
echo "  Property     : $PAIRTUNE_PROP"
echo "  Epochs       : $PAIRTUNE_EPOCHS"
echo "  Batch size   : $PAIRTUNE_BATCH"
echo "  Job          : ${SLURM_JOB_ID:-local}"
echo ""

apptainer exec \
    --rocm \
    --bind "$GPMOL_WORK_DIR":/workspace \
    --bind "$SCRIPT_DIR":/scripts \
    --bind "${GPMOL_PKGDIR}:/opt/gpmol-pkgs" \
    --env PAIRTUNE_PROP="$PAIRTUNE_PROP" \
    --env PAIRTUNE_EPOCHS="$PAIRTUNE_EPOCHS" \
    --env PAIRTUNE_BATCH="$PAIRTUNE_BATCH" \
    --env PAIRTUNE_EVAL_FREQ="$PAIRTUNE_EVAL_FREQ" \
    --env STUDIO_RESULT_OUT="${STUDIO_RESULT_OUT:-}" \
    "$GPMOL_SIF" \
    bash -c '
set -euo pipefail
cd /workspace

# Clone repo on first run
if [[ ! -d gp-molformer ]]; then
    echo "--- Cloning IBM/gp-molformer ---"
    git clone https://github.com/IBM/gp-molformer.git
fi

cd gp-molformer

# Apply compatibility patch (fixes transformers version issues)
if [[ -f /scripts/../pairtune_training.patch ]] && [[ ! -f .patched ]]; then
    git apply /scripts/../pairtune_training.patch 2>/dev/null && touch .patched \
        || echo "  (patch already applied or not needed)"
fi

echo "--- Installing dependencies ---"
pip install -q --no-cache-dir --target /opt/gpmol-pkgs \
    "accelerate==0.26.1" "datasets==2.20.0" "networkx>=3.1" \
    "numpy<2" "pandas>=2.2" "peft==0.10.0" "scikit-learn>=1.5" \
    "transformers>=4.36,<4.41" 2>&1 | tail -5
# RDKit is needed for property scoring in pair-tuning
pip install -q --no-cache-dir --target /opt/gpmol-pkgs rdkit-pypi 2>&1 | tail -3 \
    || pip install -q --no-cache-dir --target /opt/gpmol-pkgs rdkit 2>&1 | tail -3

echo "--- Stripping torch / nvidia / triton ---"
for pkg in torch torchvision torchaudio nvidia triton; do
    rm -rf "/opt/gpmol-pkgs/${pkg}" "/opt/gpmol-pkgs/${pkg}"-*.dist-info 2>/dev/null || true
done

export PYTHONPATH="/opt/gpmol-pkgs:${PYTHONPATH:-}"

GPU_OK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "$GPU_OK" != "True" ]]; then
    echo "WARNING: torch.cuda.is_available() = False — pair-tuning will use CPU (slow)." >&2
fi

# Run pair-tuning; redirect stdout so Studio can harvest per-epoch metrics
echo "--- Starting pair-tuning: property=${PAIRTUNE_PROP}, epochs=${PAIRTUNE_EPOCHS} ---"
python3 -m scripts.pairtune_training "$PAIRTUNE_PROP" \
    --num_epochs "$PAIRTUNE_EPOCHS" \
    --batch_size "$PAIRTUNE_BATCH" \
    --eval_epochs "$PAIRTUNE_EVAL_FREQ" 2>&1 | tee /tmp/pairtune_output.txt

ADAPTER_DIR="models/pairtune/${PAIRTUNE_PROP}"
echo "--- Pair-tuning done. Adapter in: /workspace/gp-molformer/${ADAPTER_DIR} ---"

# Generate molecules BEFORE and AFTER tuning (for before/after comparison)
echo "--- Generating 20 molecules (pre-adapter baseline) ---"
SCAFFOLD="" NUM_BATCHES=1 OUTPUT_FILE=/tmp/generated_before.csv \
    bash /scripts/run_generation.sh 2>&1 | tail -5

echo "--- Generating 20 molecules (post-adapter) ---"
# TODO: pass adapter dir to generation if generation script supports --adapter_path
SCAFFOLD="" NUM_BATCHES=1 OUTPUT_FILE=/tmp/generated_after.csv \
    bash /scripts/run_generation.sh 2>&1 | tail -5

echo "=== Pair-tuning complete ==="
ls -lh "/workspace/gp-molformer/${ADAPTER_DIR}/" 2>/dev/null || echo "  (adapter dir not found)"

# Write Studio result JSON if requested
if [[ -n "${STUDIO_RESULT_OUT:-}" ]]; then
    python3 - <<PYEOF
import json, os, csv, re
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors, Crippen
    def compute_props(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "smiles": smiles,
            "qed": round(QED.qed(mol), 4),
            "logp": round(Crippen.MolLogP(mol), 4),
            "mw": round(Descriptors.MolWt(mol), 2),
            "lipinski": int(Descriptors.MolWt(mol) <= 500 and
                           Crippen.MolLogP(mol) <= 5),
        }
except ImportError:
    def compute_props(smiles):
        return {"smiles": smiles, "qed": None, "logp": None, "mw": None, "lipinski": None}

def read_csv_smiles(path):
    mols = []
    try:
        with open(path) as f:
            for row in csv.reader(f):
                s = row[0].strip() if row else ""
                if s and s != "smiles":
                    p = compute_props(s)
                    if p:
                        mols.append(p)
    except Exception:
        pass
    return mols[:20]

before = read_csv_smiles("/tmp/generated_before.csv")
after  = read_csv_smiles("/tmp/generated_after.csv")

def mean_or_none(lst, key):
    vals = [x[key] for x in lst if x.get(key) is not None]
    return round(sum(vals)/len(vals), 4) if vals else None

# Parse per-epoch losses from the training log
epochs = []
for line in open("/tmp/pairtune_output.txt"):
    m = re.search(r"[Ee]poch[: ]+(\d+).*[Ll]oss[: ]+([\d.]+)", line)
    if m:
        epochs.append({"ep": int(m.group(1)), "loss": float(m.group(2))})

result = {
    "type": "molecule_finetune",
    "model": "GP-MoLFormer (pair-tuning, IBM Research)",
    "property": os.environ["PAIRTUNE_PROP"],
    "num_epochs": int(os.environ["PAIRTUNE_EPOCHS"]),
    "epochs": epochs,
    "before": {
        "qed_mean": mean_or_none(before, "qed"),
        "logp_mean": mean_or_none(before, "logp"),
        "lipinski_pass_rate": mean_or_none(before, "lipinski"),
        "molecules": before,
    },
    "after": {
        "qed_mean": mean_or_none(after, "qed"),
        "logp_mean": mean_or_none(after, "logp"),
        "lipinski_pass_rate": mean_or_none(after, "lipinski"),
        "molecules": after,
    },
}

out = os.environ["STUDIO_RESULT_OUT"]
Path(out).write_text(json.dumps(result))
print(f"[studio-result] wrote {out}")
PYEOF
fi
'

echo ""
echo "=== Done at $(date) ==="
