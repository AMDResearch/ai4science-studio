# ORBIT-2 Performance Analysis with AMD AI Agents

Multi-subagent workflow for 2-node ORBIT-2 training (`intermediate_downscaling.py`) on AMD MI355X, with Omnistat telemetry and optional rank-0 PyTorch traces.

> **Audience:** performance engineers. Output is a diagnosis of the run, not a scientific ORBIT-2 claim.

> **Data mode:** 10.0_arcmin PRISM same-dir (`superres_mag: 1`) — timing/scaling only; loss is not meaningful without 2.5_arcmin targets.

## Quick start

```bash
export AI4S_SHARED_DIR=/path/to/shared
export OMNIHUB_TOOLS_DIR=/path/to/omnihub/tools
export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/prism/10.0_arcmin
sbatch earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh
```

After the job completes, drive analyst + verifier + synthesizer subagents per [`.cursor/skills/ai4science-perf-analysis/SKILL.md`](../../../../.cursor/skills/ai4science-perf-analysis/SKILL.md).

## Artifacts

```
$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>/
├── manifest.json
├── omnistat-db/
├── omnistat.config
├── traces/orbit2-epoch*-rank0.pt.trace.json
├── orbit2-train-<jobid>.out
├── foms.json                    # from run_fom_extractor.py
├── tracelens/                   # analyst outputs
└── combined_report.md           # synthesizer output
```

## Primary FOM

**`batch_time_s`** — mean steady-state batch wall time from rank-0 log lines `Batch N: X seconds` (see `examples/parse_training_log.py`).

## Agent prompts

- [agents/launcher.md](agents/launcher.md)
- [agents/tracelens_analyst.md](agents/tracelens_analyst.md)
- [agents/omnistat_analyst.md](agents/omnistat_analyst.md)
- [agents/synthesizer.md](agents/synthesizer.md)

See [HANDOFF.md](HANDOFF.md) for validated landmines and current cluster state.
