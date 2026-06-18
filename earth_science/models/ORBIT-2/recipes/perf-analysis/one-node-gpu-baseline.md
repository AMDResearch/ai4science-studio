# ORBIT-2 — one-node GPU saturation baseline (before sysopt)

Goal: on **one MI355-class node × 8 GPUs**, pick a **per-rank batch size** and confirm the workload **uses HBM and compute** well enough to justify the optimization loop. This doc ties together **ORBIT logs**, **Omnistat**, and **TraceLens**.

**Frozen baseline checklist + VRAM-first sweep order:** [BASELINE_LOCKIN.md](BASELINE_LOCKIN.md) (update when you promote a new reference job).

> **Not a substitute for scientific validation** — same-dir PRISM/ERA5 configs are timing-friendly; loss curves are crash detectors only.

## 1. What to record (baseline table)

| Field | Source | Notes |
|--------|--------|--------|
| `trainer.batch_size` | Rendered job YAML (`rendered_config` in `manifest.json`) | Per-rank batch; global batch ≈ `batch_size × fsdp × simple_ddp` |
| `trainer.data_type` | Same YAML | `bfloat16` (Studio train/perf sbatch default) vs `float32` → **~2 vs ~4 bytes/param** for *parameter* memory only; activations still need log/Omnistat |
| Model YAML block (`preset`, `embed_dim`, `depth`, …) | Same YAML | Architecture fingerprint |
| **Parameter count** | Rank-0 log **or** one-shot container count (§4) | Needed for “model size” |
| **Parameter bytes (approx.)** | `num_params × bytes_per_element` | Weight memory order-of-magnitude; activations are extra |
| `steady_batch_time_s`, `loss_sanity_pass` | `examples/run_fom_extractor.py` → `foms.json` | Primary latency FOM; **throughput** (`throughput_samples_per_s`) when `manifest.json` has `global_batch_size` |
| GPU busy / HBM / MFMA | Omnistat VM + PromQL (see `agents/omnistat_analyst.md`) | Saturation evidence |
| Kernel timeline / overlap | TraceLens on `traces/*.pt.trace.json` | See `agents/tracelens_analyst.md` |

## 2. Submit a perf job (1 node)

**Studio default (`sbatch_train_perf_amd.sh`):** staged **ERA5 1.0°** + template **`edm_8m_era5_1x8.yaml`** (Bayes-CAST EDM; **`fsdp=8`**, **`simple_ddp=1`** after render). **`ORBIT2_ROOT`** defaults to **`…/code/bayes-cast`** when that directory exists. Override `ORBIT2_CONFIG_TEMPLATE` for **`interm_8m_era5.yaml`**. Set **`ORBIT2_ERA5_SPATIAL_RES`** if your ERA5_1 grid token count ≠ 111.

```bash
export AI4S_SHARED_DIR=...
export PERF_TOOLS_DIR=...
export ORBIT2_ROOT=...   # optional; defaults to $AI4S_SHARED_DIR/models/ORBIT-2/code/ORBIT-2
# Defaults pick ERA5 path + template; omit these if data is already staged there:
# export ORBIT2_DATA_ROOT=$AI4S_SHARED_DIR/models/ORBIT-2/data/superres/era5/1.0_deg
# export ORBIT2_CONFIG_TEMPLATE=interm_8m_era5.yaml
# Push batch until VRAM is high *and* steady_batch_time_s stops improving (not “max batch always best”):
export ORBIT2_BATCH_SIZE=4                # sweep 2, 4, 8, …
export ORBIT2_MAX_EPOCH=6
export ORBIT2_MAX_BATCHES=20
sbatch earth_science/models/ORBIT-2/examples/sbatch_train_perf_amd.sh
```

One-liner from repo root (same defaults): `earth_science/models/ORBIT-2/examples/submit_perf_baseline_era5_amd.sh`

Default parallelism is **`fsdp=8`, `simple_ddp=1`** for 1×8 (see [HANDOFF.md](HANDOFF.md)). Profiler + Omnistat start automatically.

After the job finishes:

```bash
python3 earth_science/models/ORBIT-2/examples/run_fom_extractor.py --job-dir "$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>"
python3 earth_science/models/ORBIT-2/examples/report_orbit2_gpu_baseline.py --job-dir "$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>"
```

The second command writes **`baseline_report.md`** and **`baseline_report.json`** next to `manifest.json`.

## 3. Maximizing “GPU utilization” (what we mean)

### Model size vs activation size (why ~9M params can still look “tiny”)

ORBIT-2 Studio templates (`interm_8m_prism.yaml`, `interm_8m_era5.yaml`) use **`preset: res_slimvit`** with **`embed_dim: 256`**, **`depth: 6`**, **`decoder_depth: 4`**, **`mlp_ratio: 4`**, **`num_heads: 4`**. That yields on the order of **~10M trainable parameters** — expected, not a mis-parse. **Parameter count is not the same as per-step FLOPs or VRAM:** most HBM during training is often **activations**, which scale roughly with **batch × spatial tokens × width × depth**. The PRISM same-dir template uses **`spatial_resolution: { 'PRISM': 18 }`** (a **18×18** latent grid) — very few tokens per sample, so GPUs can show **low MFMA / “low utilization”** even when the job is “correct.”

**Practical order to push toward steady-state saturation:**

1. **`ORBIT2_BATCH_SIZE` (per-rank)** — Cheapest knob; raise until you approach VRAM limits (watch Omnistat / `amd-smi` during **epoch ≥ 2**). FSDP (`fsdp=8`) shards **weights**, not activations across the sample batch in the same way; large batches still grow **per-rank** activation memory.
2. **Larger spatial workload without inventing a new architecture** — Prefer **`interm_8m_era5.yaml`** + staged ERA5 data: **`spatial_resolution: { 'ERA5_2': 111 }`** → many more tokens per forward than PRISM-18, often better for **GPU fill** and more honest perf baselines (still same-dir = non-physical loss).
3. **Widen / deepen the ViT stack (YAML `model:` block)** — Increase **`embed_dim`** and **`depth`** (and usually **`decoder_depth`**) together; keep **`num_heads`** such that **`embed_dim % num_heads == 0`**. **`mlp_ratio`** bumps FFN FLOPs and memory. These changes are **upstream-architecture-sensitive**: stay within combinations your ORBIT-2 / Bayes-CAST revision actually constructs (if training fails, compare against upstream example configs for valid presets).
4. **`superres_mag: 4`** (vs `1` in same-dir perf templates) — Restores the **high→low upsampling** path used in the paper-style setup; **strongly increases decoder-side compute and memory** but requires **consistent low/high data layout** (not the current same-dir PRISM/ERA5 perf shortcut). Use for “stress the pipeline” only when data paths match upstream expectations.
5. **`preset`** — If upstream ships additional presets (names differ by repo/version), switching preset is the largest architectural jump; treat it like a **new model** for perf baselines (re-document param count, batch, and loss sanity expectations).

After any architecture or resolution change, re-run **`steady_batch_time_s`** + **Omnistat** + **TraceLens**; do not assume a larger parameter count implies higher MFMA until you see counters during steady epochs.

**How to read saturation from telemetry**

- **HBM footprint (capacity)** — `amd-smi` / Omnistat memory series during steady epochs; increase `ORBIT2_BATCH_SIZE` until you are near the **practical** VRAM limit (leave headroom for fragmentation).
- **HBM bandwidth / MFMA** — Omnistat `rocprofiler`-backed counters (see [omnistat.config.template](omnistat.config.template)): `FETCH_SIZE` + VALU counters → interpret as compute vs bandwidth bound with TraceLens.
- **Do not equate** “largest batch that fits” with “fastest step” — watch **`steady_batch_time_s`** from logs when comparing batch sizes.

## 4. Parameter count (exact)

Upstream may not print `Total params` on every run. Use **one** of:

1. **Rank-0 log** — search for `Total model parameters:` (e.g. `9.45M`), `Total params`, `trainable params`, or `Model parameters` after a short run; then re-run the report with `--num-params N`:
   ```bash
   python3 earth_science/models/ORBIT-2/examples/report_orbit2_gpu_baseline.py \
     --job-dir "$AI4S_SHARED_DIR/models/ORBIT-2/perf-runs/<jobid>" --num-params 12345678
   ```
2. **Bayes-CAST / ORBIT helper** — if upstream documents a “print model size” snippet, run it **inside the same Apptainer image + overlay** as training, with `PYTHONPATH` matching `sbatch_train_amd.sh`, and paste the integer into `baseline_report.md` or pass `--num-params`.

Until `num_parameters` is known, `baseline_report.md` still documents **batch**, **dtype**, **YAML fingerprint**, and **FOMs** for batch sweeps.

## 5. TraceLens (overlap / kernels)

Use the perf-analysis agent flow in [`.cursor/skills/ai4science-perf-analysis/SKILL.md`](../../../../.cursor/skills/ai4science-perf-analysis/SKILL.md): `tracelens_analyst` consumes `traces/orbit2-epoch*-rank0.pt.trace.json` and writes `tracelens/report_summary.md`. Attach that summary to the baseline folder when reviewing batch sweeps.

## 6. Omnistat (HBM / XGMI / counters)

Follow [agents/omnistat_analyst.md](agents/omnistat_analyst.md): load `omnistat-db` with VictoriaMetrics (`-fs.disableMmap` on login nodes), run `omnistat-inspect` phases, and keep `omnistat/report_summary.md` beside the perf run. Use **job-windowed** PromQL (see HydraGNN perf-analysis lessons: instant queries without `time=` can be empty).

## 7. Exit criteria (“happy with initial setup”)

- **Batch size** chosen with documented **VRAM**, **steady_batch_time_s**, and **one** Omnistat + TraceLens snapshot showing no obvious stall (I/O, dataloader, or pure idle).
- **`baseline_report.md`** filled with **param count**, **param bytes estimate**, **batch**, **parallelism**, and links/paths to **foms.json**, **tracelens/**, **omnistat/**.
- Then open the broader **sysopt / perf-optimizer-loop** plan on top of this frozen baseline.
