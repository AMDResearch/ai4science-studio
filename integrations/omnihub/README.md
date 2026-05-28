# OmniHub integration for AI4Science Studio

Bridge between [AI4Science Studio](https://github.com/AMDResearch/ai4science-studio) model recipes and [OmniHub](https://github.com/AMDResearch/omnihub) job generation, without merging the two repositories.

## Prerequisites

- OmniHub clone (sibling repo), e.g. `$HOME/git/omnihub`
- `.cluster-config.yaml` with `omnihub:` section (see `.cluster-config.example.yaml`)
- `AI4S_SHARED_DIR` set (HydraGNN assets: SIF, overlay, datasets, code)
- Lux/Vultr: OmniHub `config/vultr.yaml` (`shared-dir: /shared/omnihub`)

## Quick start (HydraGNN train)

```bash
export AI4S_SHARED_DIR=/shared/$USER
export OMNIHUB_DIR=$HOME/git/omnihub

# 1. Sync application stubs into the OmniHub checkout
./integrations/omnihub/sync-to-omnihub.sh

# 2. Generate SLURM job (wrapper sets HydraGNN binds/overlay)
./integrations/omnihub/generate-job.sh \
  --num-nodes 2 --partition lux --time-limit 2h \
  --output /tmp/hydragnn-omnihub.slurm

# 3. Submit
sbatch -A vultr_lux /tmp/hydragnn-omnihub.slurm

# 4. Post-process (after job completes)
$OMNIHUB_DIR/omnihub-process --results-dir /shared/$USER/results/omnihub -j 4
$OMNIHUB_DIR/omnihub-index --results-dir /shared/$USER/results/omnihub --output index
```

## Layout

```
integrations/omnihub/
├── README.md                 # this file
├── sync-to-omnihub.sh        # rsync apps → $OMNIHUB_DIR/applications/
├── render-cluster-config.sh  # validate .cluster-config vs vultr.yaml
├── generate-job.sh           # sync + omnihub-generate-job + HydraGNN apptainer patches
├── omnistat-parity-check.sh  # compare /shared/omnihub vs omnistat-pr271
└── applications/
    └── hydragnn-train/       # OmniHub app (synced into omnihub repo)
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OMNIHUB_DIR` | Path to OmniHub repo (required for sync/generate) |
| `AI4S_SHARED_DIR` | Shared assets root (SIF, HydraGNN overlay, datasets) |
| `AI4S_REPO_ROOT` | ai4science-studio root (default: auto-detected) |

HydraGNN training knobs (`HG_*`, `HYDRAGNN_*`) match the standard `sbatch_train_amd.sh` recipe.

## Results layout

OmniHub jobs write to `/shared/$USER/results/omnihub/$SLURM_JOB_ID/` (per `config/vultr.yaml`):

```
<job_id>/
├── job.yaml, app.yaml, job-status.yaml
├── logs/srun-*.out
├── tools/omnihub-monitor/
└── processed-data/          # after omnihub-process
```

Agents should read `processed-data/` first (token-efficient). See `.cursor/skills/ai4science-omnihub/SKILL.md`.

## Recipes

| task | OmniHub tools | Notes |
|------|---------------|-------|
| `train-omnihub` | `omnihub-monitor` | Phase 1 pilot |
| `perf-analysis-omnihub` | `omnihub-monitor omnistat pytorch-trace tracelens` | Phase 2; optional `omnistat-rocprofiler-pmc1` |

Legacy HydraGNN perf (`sbatch_train_perf_amd.sh` + `omnistat-pr271`) remains available for deep agent workflows.
