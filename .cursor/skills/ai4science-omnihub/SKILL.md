---
name: ai4science-omnihub
description: Launch AI4Science Studio models via OmniHub job generation, SLURM submission, and standardized result processing. Use when the user asks for OmniHub, train-omnihub, perf-analysis-omnihub, omnihub-generate-job, or post-processing OmniHub results for ai4science models.
---

# AI4Science Studio + OmniHub

Integrates this repo with a sibling [OmniHub](https://github.com/AMDResearch/omnihub) checkout **without merging repos**.

## Prerequisites

1. Read `.cluster-config.yaml` (or `~/.config/ai4science-studio/cluster.yaml`) — especially `omnihub:` and `paths`.
2. Set `export OMNIHUB_DIR=/path/to/omnihub` and `export AI4S_SHARED_DIR=/shared/$USER` (or your site path).
3. HydraGNN assets: SIF, overlay, datasets, code clone under `$AI4S_SHARED_DIR/models/HydraGNN/`.

> **Omnistat note:** OmniHub's `/shared/omnihub/tools/omnistat` has no `omnistat-inspect`, so don't expect inspect JSON from OmniHub jobs — read `omnihub-process` output under `processed-data/` instead. For deep `omnistat-inspect`/PromQL workflows use the `omnihub-inspect` build at `/shared/omnihub/tools/omnihub-inspect` (see perf-analysis recipe).

## Workflow: HydraGNN train-omnihub

1. Read `material_science/models/HydraGNN/model.yaml` recipe `train-omnihub`.
2. Sync app stubs and generate SLURM script:

```bash
./integrations/omnihub/generate-job.sh \
  --num-nodes 2 \
  --partition lux \
  --time-limit 2h \
  --output /tmp/hydragnn-omnihub.slurm
```

3. Submit (account on `sbatch`, not in the generated script):

```bash
sbatch -A vultr_lux /tmp/hydragnn-omnihub.slurm
```

4. Monitor: `squeue -j <jobid>`, `tail -f /shared/$USER/results/omnihub/<jobid>/*.out`

5. After completion — post-process:

```bash
$OMNIHUB_DIR/omnihub-process --results-dir /shared/$USER/results/omnihub -j 4
$OMNIHUB_DIR/omnihub-index --results-dir /shared/$USER/results/omnihub --output index
```

## Workflow: perf-analysis-omnihub (Phase 2)

Same as train, but add `--perf` to enable `omnistat`, `pytorch-trace`, and `tracelens`:

```bash
./integrations/omnihub/generate-job.sh --perf --num-nodes 2 --time-limit 30m \
  --output /tmp/hydragnn-perf.slurm
sbatch -A vultr_lux /tmp/hydragnn-perf.slurm
```

**Agent result reading order** (minimal tokens):

1. `<jobdir>/processed-data/tracelens-summary.json`
2. `<jobdir>/processed-data/omnistat-range.yaml`
3. `<jobdir>/processed-data/report-card.yaml`
4. Raw traces under `<jobdir>/tools/` only if needed

Legacy HydraGNN perf (`sbatch_train_perf_amd.sh` + `omnihub-inspect` + subagent claims) remains for deep PromQL/inspect workflows. See `material_science/models/HydraGNN/recipes/perf-analysis/README.md`.

## Key paths (Lux / vultr)

| Resource | Path |
|----------|------|
| OmniHub shared | `/shared/omnihub` |
| Omnistat | `/shared/omnihub/tools/omnistat/` → use `--tools omnistat` |
| Omnistat + PMC | `/shared/omnihub/tools/omnistat-rocprofiler/` → use `--tools omnistat-rocprofiler-pmc1` for hardware counters |
| Omnistat inspect | `/shared/omnihub/tools/omnihub-inspect/` → has `omnistat-inspect` + TraceLens CLI (legacy deep-dive venv) |
| Results | `/shared/$USER/results/omnihub/$SLURM_JOB_ID/` |
| HydraGNN overlay | `$AI4S_SHARED_DIR/models/HydraGNN/overlays/hydragnn-overlay.img` |

## Runner

HydraGNN uses MPI — default `--runner manual-mpi --tasks-per-node 8` (OmniHub fork). Do **not** use `--runner manual` (that enables torch DDP init).

## Related OmniHub skills

When working inside `$OMNIHUB_DIR`, also use:

- `.cursor/skills/generate-job-and-run/SKILL.md`
- `.cursor/skills/post-process-results/SKILL.md`
- `.cursor/skills/check-job-results/SKILL.md`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| App config not found | Run `sync-to-omnihub.sh` first |
| ROCm / overlay missing | Verify `HG_OVERLAY`, `HG_SIF`; check generate-job injected binds |
| Omnistat empty | Confirm `/shared/omnihub/tools/omnistat/` exists and VictoriaMetrics binary is present |
| TraceLens skipped | Install venv at `/shared/omnihub/tools/tracelens/venv` or set `TRACELENS_VENV` |
