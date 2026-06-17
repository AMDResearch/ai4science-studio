# ORBIT-2 Performance Analysis with AMD AI Agents

Multi-subagent workflow for ORBIT-2 / Bayes-CAST training on AMD MI355X, with Omnistat telemetry and optional rank-0 PyTorch traces.

> **Audience:** performance engineers. Output is a diagnosis of the run, not a scientific ORBIT-2 claim.

> **Data mode:** Perf defaults **Bayes-CAST** **`edm_8m_era5_1x8.yaml`** + staged **ERA5 1.0°** (`ORBIT2_DATA_ROOT`). Lux **`interm_8m_lux_era5.yaml`** remains for public ORBIT-2 same-dir `res_slimvit`. Loss is not meaningful without true downscaling targets.

## Quick start (1 node × 8 GPUs, Bayes-CAST EDM template)

```bash
export AI4S_SHARED_DIR=/path/to/shared
export OMNIHUB_TOOLS_DIR=/shared/omnihub/tools
# ORBIT2_ROOT defaults to .../code/bayes-cast when that directory exists
export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/1.0_deg
# export ORBIT2_CONFIG_TEMPLATE=edm_8m_era5_1x8.yaml   # default
sbatch earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh
```

Multi-node perf: `sbatch --nodes=2 ... earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh` (requires `.cluster-config.yaml` RCCL keys). Default `#SBATCH` is **1 node**.

**GPU saturation + model/batch baseline (before sysopt):** [one-node-gpu-baseline.md](one-node-gpu-baseline.md), [BASELINE_LOCKIN.md](BASELINE_LOCKIN.md) (VRAM-first batch sweep order), and `examples/report_orbit2_gpu_baseline.py`.

After the job completes, drive analyst + verifier + synthesizer subagents per [`.cursor/skills/ai4science-perf-analysis/SKILL.md`](../../../../.cursor/skills/ai4science-perf-analysis/SKILL.md).

## Artifacts

```
$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>/
├── manifest.json              # includes git_sha, parallelism, rendered_config
├── omnistat-db/
├── omnistat.config
├── traces/orbit2-epoch*-rank0.pt.trace.json
├── orbit2-train-<jobid>.out
├── foms.json                  # from run_fom_extractor.py
├── baseline_report.md         # from report_orbit2_gpu_baseline.py (optional)
├── baseline_report.json
├── tracelens/                 # analyst outputs
└── combined_report.md         # synthesizer output
```

## Primary FOM

**`steady_batch_time_s`** — see [HANDOFF.md](HANDOFF.md) and `examples/parse_training_log.py` (epoch ≥ 2, skip warmup batches). Legacy **`batch_time_s`** label in older notes maps to the same parser options.

## Agent prompts

- [agents/launcher.md](agents/launcher.md)
- [agents/tracelens_analyst.md](agents/tracelens_analyst.md)
- [agents/omnistat_analyst.md](agents/omnistat_analyst.md)
- [agents/synthesizer.md](agents/synthesizer.md)

See [HANDOFF.md](HANDOFF.md) for validated landmines, Bayes-CAST / `launch_diffusion.sh`, and parallelism defaults.
