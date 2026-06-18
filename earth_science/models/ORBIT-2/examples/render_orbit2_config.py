#!/usr/bin/env python3
"""Render a per-job ORBIT-2 training YAML from a Studio template.

Templates include same-dir (**``interm_8m_prism.yaml``** / **``interm_8m_era5.yaml``**) and Bayes-CAST EDM
(**``edm_8m_era5_1x8.yaml``**), which also substitutes **``__ERA5_1_SPATIAL_RES__``**
from env **``ORBIT2_ERA5_SPATIAL_RES``** (default ``111`` for 1.0° ERA5 staging).

Substitutes parallelism (fsdp × simple_ddp = nodes × gpus_per_node by default), data root,
and trainer caps. ``__NUM_WORKERS__`` comes from env **``ORBIT2_NUM_WORKERS``** (default ``2``).

Env overrides (when ``--fsdp`` / ``--simple-ddp`` are omitted): ``ORBIT2_FSDP``,
``ORBIT2_SIMPLE_DDP``. ``ORBIT2_DATA_TYPE`` defaults to ``bfloat16`` when unset.

Usage:
    python3 render_orbit2_config.py --nodes 2 --output /tmp/orbit2_config.yaml
    python3 render_orbit2_config.py --nodes 8 \\
        --data-root $AI4S_SHARED_DIR/models/ORBIT-2/data/superres/prism/10.0_arcmin \\
        --max-epochs 6 --batch-size 8 -o config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ORBIT-2 training config.")
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Template YAML (default: interm_8m_prism.yaml next to this script)",
    )
    parser.add_argument("--nodes", type=int, required=True, help="SLURM node count")
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=8,
        help="GPUs per node (default: 8 for MI355X)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="PRISM data root (default: env ORBIT2_DATA_ROOT)",
    )
    parser.add_argument("--max-epochs", type=int, default=None, help="trainer.max_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="trainer.batch_size")
    parser.add_argument("--fsdp", type=int, default=None, help="Override fsdp (default: nodes)")
    parser.add_argument(
        "--simple-ddp",
        type=int,
        default=None,
        help="Override simple_ddp (default: gpus_per_node)",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output YAML path")
    args = parser.parse_args()

    import os

    script_dir = Path(__file__).resolve().parent
    template = args.template or (script_dir / "interm_8m_prism.yaml")
    if not template.is_file():
        print(f"error: template not found: {template}", file=sys.stderr)
        return 2

    data_root = args.data_root or os.environ.get("ORBIT2_DATA_ROOT")
    if not data_root:
        print("error: set --data-root or ORBIT2_DATA_ROOT", file=sys.stderr)
        return 2

    data_path = Path(data_root).resolve()
    if not data_path.is_dir():
        print(f"error: data root not found: {data_path}", file=sys.stderr)
        return 2

    # CLI overrides env; env overrides geometric default (nodes × gpus_per_node).
    fsdp = args.fsdp
    if fsdp is None and os.environ.get("ORBIT2_FSDP"):
        fsdp = int(os.environ["ORBIT2_FSDP"])
    if fsdp is None:
        fsdp = args.nodes

    simple_ddp = args.simple_ddp
    if simple_ddp is None and os.environ.get("ORBIT2_SIMPLE_DDP"):
        simple_ddp = int(os.environ["ORBIT2_SIMPLE_DDP"])
    if simple_ddp is None:
        simple_ddp = args.gpus_per_node
    total = fsdp * simple_ddp
    expected = args.nodes * args.gpus_per_node
    if total != expected:
        print(
            f"error: fsdp({fsdp}) × simple_ddp({simple_ddp}) = {total}, "
            f"expected {expected} (= nodes × gpus_per_node)",
            file=sys.stderr,
        )
        return 2

    max_epochs = args.max_epochs
    if max_epochs is None:
        max_epochs = int(os.environ.get("ORBIT2_MAX_EPOCH", "3"))
    if max_epochs < 2:
        print(
            "error: max_epochs must be >= 2 (upstream trains while epoch_start+1 < max_epochs)",
            file=sys.stderr,
        )
        return 2

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(os.environ.get("ORBIT2_BATCH_SIZE", "8"))

    data_type = os.environ.get("ORBIT2_DATA_TYPE", "bfloat16")
    if data_type not in ("float32", "bfloat16"):
        print("error: ORBIT2_DATA_TYPE must be float32 or bfloat16", file=sys.stderr)
        return 2

    # Bayes-CAST `edm_8m_era5_1x8.yaml`: spatial token count for ERA5_1 (111 ≈ 1.0° ERA5 staging)
    era5_1_spatial = int(os.environ.get("ORBIT2_ERA5_SPATIAL_RES", "111"))
    num_workers = int(os.environ.get("ORBIT2_NUM_WORKERS", "2"))

    text = template.read_text(encoding="utf-8")
    replacements = {
        "__DATA_ROOT__": str(data_path),
        "__FSDP__": str(fsdp),
        "__SIMPLE_DDP__": str(simple_ddp),
        "__MAX_EPOCHS__": str(max_epochs),
        "__BATCH_SIZE__": str(batch_size),
        "__DATA_TYPE__": data_type,
        "__ERA5_1_SPATIAL_RES__": str(era5_1_spatial),
        "__NUM_WORKERS__": str(num_workers),
    }
    for key, val in replacements.items():
        text = text.replace(key, val)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output} (fsdp={fsdp}, simple_ddp={simple_ddp}, epochs={max_epochs}, batch={batch_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
