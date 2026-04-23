# Check model readiness status

Audit which models are fully set up and which have gaps.

## How to respond

For each model listed in `models.yaml`, check the following files exist:

| Check | File pattern |
|-------|-------------|
| Model README | `<path>/README.md` |
| Model manifest | `<path>/model.yaml` |
| Docker script | `<path>/examples/docker_run.sh` |
| Preflight script | `<path>/examples/preflight_*.py` |
| Run script | `<path>/examples/run_*.{sh,py}` |
| SLURM script | `<path>/examples/sbatch_*_amd.sh` |
| Examples README | `<path>/examples/README.md` |
| At least one recipe | `<path>/recipes/*/README.md` |

Present the results as a checklist table:

| Model | README | model.yaml | docker | preflight | run script | SLURM | examples README | recipes |
|-------|--------|------------|--------|-----------|------------|-------|----------------|---------|
| StormCast | Y | Y | Y | Y | Y | Y | Y | 2 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

Flag any models with missing items and suggest what to add.

## Arguments

$ARGUMENTS
