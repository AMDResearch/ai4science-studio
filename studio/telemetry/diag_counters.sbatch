#!/bin/bash
#SBATCH --job-name=cnt-diag
#SBATCH --partition=lux
#SBATCH --account=vultr_lux
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:08:00
#SBATCH --output=/shared/spannala/models/HydraGNN/train_work/logs/cnt_diag_%j.log
#SBATCH --error=/shared/spannala/models/HydraGNN/train_work/logs/cnt_diag_%j.log
set -uo pipefail
VENV=/shared/spannala/perf-tools/omnistat-venv
export PATH=/opt/rocm/bin:$PATH

echo "=== node $(hostname), gfx: $(/opt/rocm/llvm/bin/amdgpu-arch 2>/dev/null | head -1) ==="

echo "=== 1. Does rocprofv3 list our counters as valid for this GPU? ==="
rocprofv3 --list-avail 2>/dev/null | grep -iE "FETCH_SIZE|SQ_INSTS_VALU_MFMA_MOPS_F64|SQ_INSTS_VALU_ADD_F64" | head

echo "=== 2. Run a tiny GPU workload under rocprofv3 with our exact counters ==="
cat > /tmp/gpu_burn.py <<'PY'
import torch, time
d = torch.device("cuda:0")
a = torch.randn(4096, 4096, dtype=torch.float64, device=d)
b = torch.randn(4096, 4096, dtype=torch.float64, device=d)
for _ in range(200):
    c = a @ b
torch.cuda.synchronize()
print("fp64 matmul done", c.sum().item())
PY
echo "FETCH_SIZE SQ_INSTS_VALU_MFMA_MOPS_F64 SQ_INSTS_VALU_ADD_F64 SQ_INSTS_VALU_MUL_F64 SQ_INSTS_VALU_FMA_F64" > /tmp/counters.txt
SIF=/shared/spannala/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
OVL=/shared/spannala/models/HydraGNN/overlays/hydragnn-overlay.img
_U=(); while IFS= read -r v; do _U+=(-u "$v"); done < <(env | grep -oE '^(PMIX_|PMI_|OMPI_)[A-Za-z0-9_]+')
srun --ntasks=1 --gpus-per-node=8 --mpi=pmix --cpu-bind=none \
  env "${_U[@]}" apptainer exec --rocm --overlay "${OVL}:ro" \
  --bind /tmp:/tmp "$SIF" bash -lc '
    source /opt/venv/bin/activate 2>/dev/null
    export PATH=/opt/rocm/bin:$PATH
    cd /tmp
    echo "--- rocprofv3 counter collection over fp64 matmul ---"
    rocprofv3 -i /tmp/counters.txt --output-format csv -d /tmp/rocprof_out -- python3 /tmp/gpu_burn.py 2>&1 | tail -8
    echo "--- resulting counter CSV (nonzero?) ---"
    find /tmp/rocprof_out -name "*counter*.csv" 2>/dev/null | head -1 | xargs head -12 2>/dev/null
  ' 2>&1 | tail -40
echo "=== END CNT DIAG ==="
