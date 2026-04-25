# ArchesWeather — Training Recipe

Train ArchesWeather (deterministic) and ArchesWeatherGen (generative) on ERA5 reanalysis data.

> **Ready-to-run scripts** live in [`../../examples/`](../../examples/).
> Use [`run_train.sh`](../../examples/run_train.sh) directly instead of
> copying snippets from this doc. The SLURM driver is
> [`sbatch_train_amd.sh`](../../examples/sbatch_train_amd.sh).

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| **Container** | `pytorch_training_geoarches:latest` from silogen/ai-samples |
| **GPU** | AMD Instinct MI300X (192 GB HBM3) recommended |
| **Runtime** | Docker with ROCm kernel-mode driver (`amdgpu-dkms`) |
| **Data** | ~735 GB disk space for full ERA5 dataset |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL` | No | `archesweather` | `archesweather` or `archesweathergen` |
| `PHASE` | No | `pretrain` | `pretrain` or `finetune` |
| `SEED` | No | `0` | Seed index 0–3 for ensemble diversity |
| `PRECISION` | No | `16-mixed` | `32-true`, `16-mixed`, or `bf16-mixed` |
| `BATCH_SIZE` | No | `8` | Per-GPU batch size (5 at 32-true, 8 at 16/bf16-mixed on MI300X) |
| `MAX_STEPS` | No | Model-dependent | Training steps (250K pretrain, 50K finetune for ArchesWeather) |
| `DATA_PATH` | No | `/data/era5_240/full` | ERA5 dataset path inside container |
| `LOAD_FROM` | Finetune only | — | Checkpoint path for fine-tuning phase |

## Quick Start

```bash
bash examples/run_train.sh
```

## Setup

```bash
git clone https://github.com/silogen/ai-samples.git
cd ai4sciences/geoarches-training

docker build -t pytorch_training_geoarches:latest .
docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --shm-size=16g \
    -v $(pwd):/workspace \
    pytorch_training_geoarches bash
```

## Download ERA5 data

Inside the container:

```bash
python -m geoarches.download.dl_era.py \
    --output_dir data/era5_240/full/
```

This downloads 6 upper-air + 4 surface variables at 1.5° resolution for 1979–2020 (~735 GB).

## Train ArchesWeather (deterministic)

### Phase 1 — Pretraining (250K steps, full 1979–2018 dataset)

```bash
# Train 4 seeds for ensemble diversity
for i in {0..3}; do
    python -m geoarches.main_hydra ++log=True \
        dataloader=era5 module=archesweather \
        ++name=archesweather-m-seed$i \
        ++cluster.precision=32-true \
        ++batch_size=4 \
        ++max_steps=250000 \
        ++save_step_frequency=50000 \
        ++dataloader.dataset.path=data/era5_240/full/
done
```

### Phase 2 — Recent-past fine-tuning (50K steps, 2007–2018)

```bash
python -m geoarches.main_hydra ++log=True \
    dataloader=era5 module=archesweather \
    ++name=archesweather-m-seed0-finetuned \
    ++cluster.precision=32-true \
    ++batch_size=4 \
    ++max_steps=50000 \
    ++dataloader.dataset.start_year=2007 \
    ++dataloader.dataset.end_year=2018 \
    ++dataloader.dataset.path=data/era5_240/full/ \
    ++load_from=checkpoints/archesweather-m-seed0/
```

## Train ArchesWeatherGen (generative / flow matching)

### Phase 1 — Pretraining on residuals (200K steps)

```bash
python -m geoarches.main_hydra ++log=True \
    dataloader=era5 module=archesweathergen \
    ++name=archesweathergen \
    ++cluster.precision=32-true \
    ++batch_size=4 \
    ++max_steps=200000 \
    ++dataloader.dataset.path=data/era5_240/full/
```

### Phase 2 — OOD fine-tuning (60K steps on 2019)

```bash
python -m geoarches.main_hydra ++log=True \
    dataloader=era5 module=archesweathergen \
    ++name=archesweathergen-finetuned \
    ++cluster.precision=32-true \
    ++batch_size=4 \
    ++max_steps=60000 \
    ++dataloader.dataset.start_year=2019 \
    ++dataloader.dataset.end_year=2019 \
    ++dataloader.dataset.path=data/era5_240/full/ \
    ++load_from=checkpoints/archesweathergen/
```

## Precision and batch size

| Precision flag | Max batch size (M-model, MI300X) |
|---|---|
| `32-true` | 5 |
| `16-mixed` | 8 |
| `bf16-mixed` | 8 |

Multi-GPU training uses PyTorch DDP via PyTorch Lightning automatically.

## References

- [AMD ROCm blog — Training ArchesWeather](https://rocm.blogs.amd.com/artificial-intelligence/geoarches-training/README.html)
- [geoarches documentation](https://geoarches.readthedocs.io)
- [ArchesWeather paper](https://arxiv.org/abs/2412.12971)
