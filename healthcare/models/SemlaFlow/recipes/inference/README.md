# SemlaFlow — Inference Recipe

> **Research / engineering use only.** Not for clinical or diagnostic use.

Sample novel 3D drug-like molecular structures from a pretrained SemlaFlow checkpoint.

## Prerequisites

- AMD Instinct GPU with ROCm 6.4.1+ driver
- Docker with GPU device access
- Pretrained checkpoint (see [upstream repo](https://github.com/rssrwn/semla-flow) for Google Drive links)

## Setup

```bash
git clone https://github.com/rssrwn/semla-flow.git
cd semla-flow
```

Build the Docker image:

```dockerfile
FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0

WORKDIR /semlaflow
COPY . .
# Replace pytorch-cuda with rocm-pytorch in environment.yml before building
RUN conda env update --name base --file environment.yml
```

```bash
docker build -t rocm-semlaflow .
```

Set paths and run:

```bash
export DATA_PATH=/path/to/data
export OUTPUT_PATH=/path/to/output

docker run -it --shm-size=256g \
    --device=/dev/kfd \
    --device=/dev/dri/renderD<RENDER_ID> \
    --network host --ipc host \
    -v $DATA_PATH:/data \
    -v $OUTPUT_PATH:/output \
    rocm-semlaflow \
    predict /output/molecules.sdf --data_path /data/<dataset> <other_args>
```

## Core scripts

| Script | Purpose |
|---|---|
| `preprocess` | Prepare dataset into model representation |
| `train` | Train model on prepared data |
| `evaluate` | Assess generated structures |
| `predict` | Sample new molecules (inference) |

## torch.compile optimization (recommended)

Add `torch.compile` to the model before sampling for a ~44% throughput gain:

```python
model = torch.compile(model, dynamic=True, fullgraph=False)
```

> Set compiler cache limit to 1000 in config to prevent recompilation overhead.

## Disable EMA for faster inference

Pass `--no_ema` to skip exponential moving average tracking during sampling (~8% additional speedup):

```bash
predict /output/molecules.sdf --data_path /data/<dataset> --no_ema
```

## Datasets

SemlaFlow is trained and benchmarked on:

| Dataset | Description |
|---|---|
| QM9 | ~134k small organic molecules with quantum chemical properties |
| GEOM-Drugs | ~450k drug-like conformers from GEOM |

Pretrained checkpoints for both datasets are available via the upstream GitHub repo (Google Drive links).

## References

- [AMD ROCm blog — Part 2: SemlaFlow](https://rocm.blogs.amd.com/artificial-intelligence/semlaflow/README.html)
- [SemlaFlow paper (OpenReview)](https://openreview.net/forum?id=bee2G6pEh0)
- [GitHub repo](https://github.com/rssrwn/semla-flow)
