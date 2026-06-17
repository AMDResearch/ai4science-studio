"""Rank-0 PyTorch profiler hook for ORBIT-2 perf-analysis runs.

Set ORBIT2_RANK_PRE_TRAIN_HOOK to this file path. Env:
  ORBIT2_PROFILE_DIR     Trace output directory (default: perf-runs/<jobid>/traces)
  PROFILE_TARGET_EPOCH   Epoch index to profile (default: 0)
  PROFILE_RANK0_ONLY     If 1, only rank 0 profiles (default: 1)
"""

from __future__ import annotations

import os
from pathlib import Path

import intermediate_downscaling as ids
import torch


def _wrap_training():
    world_rank = int(os.environ.get("SLURM_PROCID", "0"))
    rank0_only = os.environ.get("PROFILE_RANK0_ONLY", "1") == "1"
    if rank0_only and world_rank != 0:
        return

    target_epoch = int(os.environ.get("PROFILE_TARGET_EPOCH", "0"))
    profile_dir = Path(
        os.environ.get(
            "ORBIT2_PROFILE_DIR",
            os.environ.get("ORBIT2_OUTPUT_DIR", "/tmp") + "/traces",
        )
    )

    _orig = ids.run_training_epochs

    def _profiled(*args, **kwargs):
        epoch_start = kwargs.get("epoch_start", args[7] if len(args) > 7 else 0)
        epoch_end = kwargs.get("epoch_end", args[8] if len(args) > 8 else epoch_start + 1)
        world_rank_kw = kwargs.get("world_rank", args[12] if len(args) > 12 else world_rank)

        if world_rank_kw != 0:
            return _orig(*args, **kwargs)

        profile_dir.mkdir(parents=True, exist_ok=True)
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

        for epoch in range(epoch_start, epoch_end):
            if epoch != target_epoch:
                # Run single epoch without profiler
                kwargs_ep = dict(kwargs)
                kwargs_ep["epoch_start"] = epoch
                kwargs_ep["epoch_end"] = epoch + 1
                _orig(*args, **kwargs_ep)
                continue

            trace_path = profile_dir / f"orbit2-epoch{epoch}-rank0"
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_path)),
            ) as prof:
                kwargs_ep = dict(kwargs)
                kwargs_ep["epoch_start"] = epoch
                kwargs_ep["epoch_end"] = epoch + 1
                result = _orig(*args, **kwargs_ep)
            # on_trace_ready already saved the trace; do not call export_chrome_trace again
            # (RuntimeError: Trace is already saved).
            return result

        return epoch_end

    ids.run_training_epochs = _profiled


_wrap_training()
