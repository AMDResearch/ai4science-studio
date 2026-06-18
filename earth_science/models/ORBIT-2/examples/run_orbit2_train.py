#!/usr/bin/env python3
"""Launch upstream ORBIT-2 intermediate_downscaling.py with optional training caps.

Patches run_training_epochs to honour ORBIT2_MAX_BATCHES (upstream has no batch cap).
Config path is passed as argv[1] after this script's own args.

Environment:
    ORBIT2_MAX_BATCHES  Cap batches per epoch (0 = unlimited)
    ORBIT2_RANK_PRE_TRAIN_HOOK  Optional path to a .py hook run before training
    ORBIT2_FUSED_ATTN=DEFAULT  Force PyTorch SDPA instead of xformers CK path
"""

from __future__ import annotations

import os
import runpy
import sys
import types
from datetime import timedelta
from pathlib import Path


def _stub_gptl4py() -> None:
    """Upstream training imports gptl4py (Frontier GPTL); stub for ROCm containers."""
    if "gptl4py" in sys.modules:
        return

    class _Stub:
        @staticmethod
        def start(_name: str) -> None:
            return None

        @staticmethod
        def stop(_name: str) -> None:
            return None

    mod = types.ModuleType("gptl4py")
    mod.start = _Stub.start
    mod.stop = _Stub.stop
    sys.modules["gptl4py"] = mod


def _patch_fused_attn_if_needed(ids) -> None:
    """Use PyTorch SDPA when overlay xformers lacks .ops (CK path unavailable)."""
    force = os.environ.get("ORBIT2_FUSED_ATTN", "").upper() == "DEFAULT"
    try:
        import xformers  # noqa: F401

        if not hasattr(xformers, "ops"):
            force = True
        else:
            import xformers.ops  # noqa: F401
    except (ImportError, AttributeError):
        force = True

    if not force:
        return

    from climate_learn.utils.fused_attn import FusedAttn

    class _PatchedFA:
        CK = FusedAttn.DEFAULT
        DEFAULT = FusedAttn.DEFAULT
        NONE = FusedAttn.NONE

    ids.FusedAttn = _PatchedFA


def _apply_batch_cap(ids) -> None:
    max_batches = int(os.environ.get("ORBIT2_MAX_BATCHES", "0") or 0)
    if max_batches <= 0:
        return

    import itertools

    _orig = ids.run_training_epochs

    def _cap_dataloader(dl, limit: int):
        class _Capped:
            def __init__(self, inner, n: int):
                self._inner = inner
                self._n = n

            def __iter__(self):
                return iter(itertools.islice(iter(self._inner), self._n))

            def __len__(self):
                try:
                    return min(len(self._inner), self._n)
                except TypeError:
                    return self._n

        return _Capped(dl, limit)

    def _capped(*args, **kwargs):
        dl = kwargs.get("train_dataloader")
        if dl is None and len(args) > 4:
            dl = args[4]
            args = list(args)
            args[4] = _cap_dataloader(dl, max_batches)
            args = tuple(args)
        elif dl is not None:
            kwargs["train_dataloader"] = _cap_dataloader(dl, max_batches)
        world_rank = kwargs.get("world_rank", args[12] if len(args) > 12 else -1)
        if world_rank == 0:
            print(f"[run_orbit2_train] ORBIT2_MAX_BATCHES cap: {max_batches}", flush=True)
        return _orig(*args, **kwargs)

    ids.run_training_epochs = _capped


def _run_hook() -> None:
    hook = os.environ.get("ORBIT2_RANK_PRE_TRAIN_HOOK", "").strip()
    if not hook:
        return
    hook_path = Path(hook)
    if not hook_path.is_file():
        raise FileNotFoundError(f"ORBIT2_RANK_PRE_TRAIN_HOOK not found: {hook}")
    # Profiler hook patches rank-0 training only; skip import on other ranks.
    rank0_only = os.environ.get("PROFILE_RANK0_ONLY", "1") == "1"
    world_rank = int(os.environ.get("SLURM_PROCID", "0"))
    if rank0_only and world_rank != 0:
        return
    runpy.run_path(str(hook_path), run_name="__orbit2_hook__")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: run_orbit2_train.py <config.yaml>", file=sys.stderr)
        sys.exit(2)

    config = sys.argv[1]
    orbit2_root = os.environ.get("ORBIT2_ROOT")
    if not orbit2_root:
        print("error: ORBIT2_ROOT must be set", file=sys.stderr)
        sys.exit(2)

    examples_dir = Path(orbit2_root) / "examples"
    if not examples_dir.is_dir():
        print(f"error: missing examples dir: {examples_dir}", file=sys.stderr)
        sys.exit(2)

    src_dir = Path(orbit2_root) / "src"
    for p in (str(src_dir), str(examples_dir), orbit2_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    os.chdir(examples_dir)
    _stub_gptl4py()

    import torch
    import torch.distributed as dist

    import intermediate_downscaling as ids  # noqa: WPS433

    _patch_fused_attn_if_needed(ids)
    _apply_batch_cap(ids)
    _run_hook()

    sys.argv = ["intermediate_downscaling.py", config]

    os.environ["MASTER_ADDR"] = str(os.environ["HOSTNAME"])
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["RANK"] = os.environ["SLURM_PROCID"]

    local_rank = int(os.environ["SLURM_LOCALID"])
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=7200000),
        rank=int(os.environ["SLURM_PROCID"]),
        world_size=int(os.environ["SLURM_NTASKS"]),
    )

    print("Using dist.init_process_group. world_size ", os.environ["SLURM_NTASKS"], flush=True)

    try:
        ids.main(device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
