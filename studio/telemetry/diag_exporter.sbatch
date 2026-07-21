#!/bin/bash
#SBATCH --job-name=tele-diag
#SBATCH --partition=lux
#SBATCH --account=vultr_lux
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=/shared/spannala/models/HydraGNN/train_work/logs/tele_diag_%j.log
#SBATCH --error=/shared/spannala/models/HydraGNN/train_work/logs/tele_diag_%j.log
set -uo pipefail
VENV=/shared/spannala/perf-tools/omnistat-venv
echo "=== node: $(hostname) ==="
echo "--- /opt/rocm present on bare node? ---"
ls -d /opt/rocm*/share/amd_smi/amdsmi 2>&1 | head
echo "--- amdsmi importable by our venv python on bare node? ---"
$VENV/bin/python - <<'PY' 2>&1 | head -20
import importlib.util, glob, sys
cands = glob.glob("/opt/rocm*/share/amd_smi/amdsmi/__init__.py")
print("amdsmi candidates:", cands)
if cands:
    d = cands[0].rsplit("/amdsmi/",1)[0]
    sys.path.insert(0, d)
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        h = amdsmi.amdsmi_get_processor_handles()
        print("amdsmi OK, num GPUs:", len(h))
        print("sample power:", amdsmi.amdsmi_get_power_info(h[0]).get("average_socket_power","?"))
        amdsmi.amdsmi_shut_down()
    except Exception as e:
        print("amdsmi ERROR:", repr(e))
else:
    print("no amdsmi found")
PY
echo "=== END DIAG ==="
