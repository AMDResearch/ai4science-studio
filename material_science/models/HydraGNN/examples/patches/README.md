# HydraGNN local patches

Small, opt-in patches applied during overlay build (see
`examples/build_overlay_amd.sh`). Each patch is unified-format and applies
cleanly with `patch -p1` against the pinned `HG_HYDRAGNN_SHA` in the build
script.

## load_data.persistent_workers.patch

**Target SHA:** `2fb0bd0157e3c85a74f9841887155095bd163303` (upstream `main`, 2026-05-20)

**Lines changed:** +4 / -0 in `hydragnn/preprocess/load_data.py`

**What it does**

Upstream already exposes `num_workers` as an env-var knob
(`HYDRAGNN_NUM_WORKERS`) but hard-codes `persistent_workers = False`. This
patch adds a companion opt-in env var:

- `HYDRAGNN_PERSISTENT_WORKERS=1` and `num_workers > 0`
  → `persistent_workers=True` is passed to all DataLoaders
- Otherwise the default upstream behaviour (`persistent_workers=False`) is
  unchanged.

**Why**

When `num_workers > 0` and `persistent_workers=False`, PyTorch forks the
worker processes at the start of every epoch and re-imports the dataset
module + re-mmaps ADIOS2 BP files. This shows up in our profiler traces as a
bimodal iteration-time distribution — most iterations at the steady-state
s/it, plus a long tail clustered around epoch boundaries and after
validation hops. Setting `persistent_workers=True` keeps the workers alive
across epochs and is PyTorch's documented cure for that pattern.

**Status**

Ad-hoc local patch for the `ai4science-studio` perf-analysis recipe. Once
its impact is measured, the right follow-up is a small upstream PR to
`ORNL/HydraGNN` adding the `HYDRAGNN_PERSISTENT_WORKERS` env knob with the
same opt-in semantics.
