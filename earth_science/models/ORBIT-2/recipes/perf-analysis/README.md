# ORBIT-2 Performance Analysis with AMD AI Agents

Multi-subagent workflow for ORBIT-2 / Bayes-CAST training on AMD MI355X, with Omnistat telemetry and optional rank-0 PyTorch traces.

> **Audience:** performance engineers. Output is a diagnosis of the run, not a scientific ORBIT-2 claim.

> **Data mode:** Perf defaults **Bayes-CAST** **`edm_8m_era5_1x8.yaml`** + staged **ERA5 1.0°** (`ORBIT2_DATA_ROOT`). **`interm_8m_era5.yaml`** remains for public ORBIT-2 same-dir `res_slimvit`. Loss is not meaningful without true downscaling targets.

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

## FOM contract (steady-state + loss sanity)

**Primary FOM:** `steady_batch_time_s` — mean per-batch wall time from **epoch ≥ 2**, excluding **batch index < 1** within each epoch (skips compile/warmup batch 0). Epochs 0–1 are warmup. Legacy **`batch_time_s`** in older notes maps to the same parser options.

**Loss sanity:** epoch-completed losses must **strictly decrease** epoch-over-epoch (`loss_sanity_pass`). Same-dir data makes absolute loss non-physical; use as a crash/hang detector only.

**Scaling runs:** use `ORBIT2_MAX_EPOCH=6` (trains epochs 0–4) so the steady window spans epochs 2–4. Parse with:

```bash
python3 examples/parse_training_log.py --log ... --steady-epoch-start 2 --warmup-batches-per-epoch 1
python3 examples/collate_scaling_study.py --log-dir .../logs --jobs ... --steady-epoch-start 2 --require-loss-sanity
```

## Defaults & landmines

- **Parallelism (1 node × 8 GPU):** `fsdp=8`, `simple_ddp=1` unless `ORBIT2_FSDP` / `ORBIT2_SIMPLE_DDP` are set; multi-node renders `fsdp=nodes`, `simple_ddp=8`.
- **Launcher discovery:** when `$ORBIT2_ROOT/launch/train_edm.py` exists (Bayes-CAST), ranks run `train_edm.py /config/config.yaml` from `/orbit2/launch`; do **not** exec upstream `launch_diffusion.sh` (OLCF/Crusher job wrapper — nested `srun`, ignores argv). For public ORBIT-2, `launch_diffusion.sh` + argv applies when `train_edm.py` is absent.
- **Same-dir data:** set `superres_mag: 1` in `interm_8m_prism.yaml` / `interm_8m_era5.yaml`, else tensor-shape mismatch.
- **`max_epochs` must be ≥ 2** (upstream loop is `while (epoch_start + 1) < max_epochs`).
- For the **GEMM-time bottleneck**, MIOpen-flag A/B, and the tabled conv-padding / dead-end `channels_last` levers, see [`../perf-optimizer-loop/gemm-attribution.md`](../perf-optimizer-loop/gemm-attribution.md). Cross-cutting lessons (bf16 Flash 65535 cap, xFormers CK, std_delta/ERA5_1, debug-log handling) live in the earth-science and perf-analysis SKILLs.

## Agent prompts

- [agents/launcher.md](agents/launcher.md)
- [agents/tracelens_analyst.md](agents/tracelens_analyst.md)
- [agents/omnistat_analyst.md](agents/omnistat_analyst.md)
- [agents/synthesizer.md](agents/synthesizer.md)
