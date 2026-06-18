#!/usr/bin/env python3
"""Launch Bayes-CAST EDM launch/train_edm.py with optional training caps.

train_edm.py runs its batch loop inline in main() (no run_training_epochs to
patch like the studio intermediate_downscaling path). So the cap is applied at
the dataloader: IterDataModule.train_dataloader is wrapped to yield at most
ORBIT2_MAX_BATCHES batches per epoch. Mirrors run_orbit2_train.py.

Config path is passed as argv[1] after this script's own args.

Environment:
    ORBIT2_ROOT         Bayes-CAST checkout (must contain launch/train_edm.py)
    ORBIT2_MAX_BATCHES  Cap batches per epoch (0 = unlimited)
"""

from __future__ import annotations

import itertools
import os
import sys
import types
from datetime import timedelta
from pathlib import Path


def _stub_gptl4py() -> None:
    """train_edm may import gptl4py (Frontier GPTL); stub for ROCm containers."""
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


def _apply_batch_cap() -> None:
    max_batches = int(os.environ.get("ORBIT2_MAX_BATCHES", "0") or 0)
    if max_batches <= 0:
        return

    from climate_learn.data.itermodule import IterDataModule

    _orig = IterDataModule.train_dataloader

    def _capped(self, *args, **kwargs):
        dl = _orig(self, *args, **kwargs)
        if int(os.getenv("SLURM_PROCID", "0")) == 0:
            print(f"[run_orbit2_train_edm] ORBIT2_MAX_BATCHES cap: {max_batches}", flush=True)
        return _cap_dataloader(dl, max_batches)

    IterDataModule.train_dataloader = _capped


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: run_orbit2_train_edm.py <config.yaml>", file=sys.stderr)
        sys.exit(2)

    config = sys.argv[1]
    orbit2_root = os.environ.get("ORBIT2_ROOT")
    if not orbit2_root:
        print("error: ORBIT2_ROOT must be set", file=sys.stderr)
        sys.exit(2)

    launch_dir = Path(orbit2_root) / "launch"
    if not (launch_dir / "train_edm.py").is_file():
        print(f"error: missing launch/train_edm.py under: {orbit2_root}", file=sys.stderr)
        sys.exit(2)

    src_dir = Path(orbit2_root) / "src"
    for p in (str(src_dir), str(launch_dir), orbit2_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    os.chdir(launch_dir)
    _stub_gptl4py()

    import torch
    import torch.distributed as dist

    import train_edm  # noqa: WPS433

    _apply_batch_cap()

    # train_edm.main() reads sys.argv[1] as the config path.
    sys.argv = ["train_edm.py", config]

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
        train_edm.main(device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
