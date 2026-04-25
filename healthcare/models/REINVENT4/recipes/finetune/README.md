# REINVENT4 — Transfer Learning (Fine-Tune) Recipe

> **Research / engineering use only.** Not for clinical or diagnostic use.

Use REINVENT4's Transfer Learning (TL) mode to bias a pretrained generative model toward a target chemical space — useful for focused library generation around a pharmacophore or hit compound series.

## Prerequisites

- Container: `rocm/pytorch:rocm6.3.3_ubuntu24.04_py3.12_pytorch_release_2.6.0`
- GPU: AMD Instinct with ROCm 6.3.3+ driver
- Runtime: Docker with GPU device access
- Data: ChEMBL35 SMILES (see download section below)

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CONFIG_FILE` | Yes | `/workspace/tl_config.toml` | TOML configuration for transfer learning |
| `RESULTS_DIR` | No | `/workspace/results` | Logs and results directory |

## Quick Start

```bash
cd healthcare/models/REINVENT4/examples
./docker_run.sh
# Then inside the container:
bash /workspace/run_tl.sh
```

## Setup

```bash
git clone https://github.com/MolecularAI/REINVENT4.git
cd REINVENT4
```

Build the container:

```dockerfile
FROM rocm/pytorch:rocm6.3.3_ubuntu22.04_py3.10_pytorch_release_2.2.1

WORKDIR /reinvent4
COPY . .
# Remove PyTorch from dep list to avoid reinstalling over ROCm image
RUN sed -i '/^torch/d; /^torchvision/d' pyproject.toml
RUN python install.py

ENTRYPOINT ["reinvent"]
```

```bash
docker build -t reinvent4-rocm .
```

Run:

```bash
docker run --device=/dev/kfd --device=/dev/dri/renderD<GPU_ID> \
    --group-add video --network host \
    -v $CONFIG_DIR:/configs -v $RESULTS_DIR:/results \
    reinvent4-rocm -l /results/run.log /configs/tl_config.toml
```

## Transfer learning config

Create a TOML config file `tl_config.toml`:

```toml
[parameters]
  mode = "TL"
  model_file = "priors/reinvent.prior"          # pretrained LSTM prior
  input_smiles_file = "data/target_smiles.smi"  # your focused SMILES set
  output_model_file = "results/finetuned.model"
  num_epochs = 10
  batch_size = 128
  learning_rate = 1e-4

[learning_rate_scheduler]
  type = "StepLR"
  step = 1
  gamma = 0.95
```

Run:

```bash
reinvent -l run.log tl_config.toml
```

## AMP optimization (recommended for AMD GPUs)

Modify `reinvent/runmodes/TL/learning.py` to enable mixed-precision training:

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# Inside the training loop:
with autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

This change delivers **10–60% faster training** with no accuracy loss.

## Dataset

Download ChEMBL35 SMILES for pretraining a new prior or assembling a focused fine-tuning set:

```bash
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_chemreps.txt.gz
gunzip chembl_35_chemreps.txt.gz
# Extract canonical SMILES column for use as input_smiles_file
awk -F'\t' 'NR>1 {print $2}' chembl_35_chemreps.txt > chembl35.smi
```

## References

- [AMD ROCm blog — Part 1: REINVENT4](https://rocm.blogs.amd.com/artificial-intelligence/running-reinvent4-amd/README.html)
- [REINVENT4 paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00812-5)
- [GitHub repo](https://github.com/MolecularAI/REINVENT4)
