# ArchesWeather — Inference Recipe

Run deterministic and ensemble weather forecasts using pretrained ArchesWeather / ArchesWeatherGen checkpoints from Hugging Face.

## Prerequisites

- AMD Instinct GPU with ROCm driver
- Docker with GPU device access

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

## Download pretrained checkpoints

Pretrained weights for all ArchesWeather seeds and ArchesWeatherGen are available on Hugging Face:

```python
from huggingface_hub import snapshot_download

# Download all ArchesWeather M-model seeds
snapshot_download(repo_id="gcouairon/ArchesWeather", local_dir="checkpoints/")
```

Available checkpoints:

| Name | Type |
|---|---|
| `archesweather-m-seed0` through `seed3` | Deterministic |
| `archesweathergen` | Generative (flow matching) |

## Deterministic inference

Pre-compute predictions across the test set (2020):

```bash
python -m geoarches.inference.encode_dataset \
    ++checkpoint_path=checkpoints/archesweather-m-seed0 \
    ++dataloader.dataset.path=data/era5_240/full/ \
    ++dataloader.dataset.start_year=2020 \
    ++dataloader.dataset.end_year=2020 \
    ++output_path=results/predictions/
```

## Ensemble inference (ArchesWeatherGen)

ArchesWeatherGen runs 25 neural-network calls per forecast via an Euler ODE solver:

```bash
python -m geoarches.inference.encode_dataset \
    module=archesweathergen \
    ++checkpoint_path=checkpoints/archesweathergen \
    ++dataloader.dataset.path=data/era5_240/full/ \
    ++dataloader.dataset.start_year=2020 \
    ++output_path=results/gen_predictions/
```

## Evaluation metrics

```bash
python -m geoarches.evaluation.evaluate \
    ++predictions_path=results/predictions/ \
    ++target_path=data/era5_240/full/ \
    ++metrics=[rmse,crps,brier_skill_score,rank_histogram]
```

## Visualization

Generate GIFs of temporal forecast evolution:

```bash
python -m geoarches.evaluation.plot \
    ++predictions_path=results/predictions/ \
    ++output_dir=results/figures/
```

## References

- [AMD ROCm blog — Training ArchesWeather on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/geoarches-training/README.html)
- [Hugging Face model card](https://huggingface.co/gcouairon/ArchesWeather)
- [geoarches documentation](https://geoarches.readthedocs.io)
