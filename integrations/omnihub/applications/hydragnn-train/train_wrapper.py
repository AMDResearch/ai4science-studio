#!/usr/bin/env python3
"""
HydraGNN training wrapper for OmniHub (AI4Science Studio integration).

Runs the upstream multidataset_hpo_sc26 entrypoint inside the container with
MPI-style rank identity from Slurm (manual-mpi runner). Mirrors the rank script
in material_science/models/HydraGNN/examples/sbatch_train_amd.sh.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import omnihub.run
import omnihub.tools

# Pinned SHA matches sbatch_train_amd.sh default.
DEFAULT_HG_SHA = "2fb0bd0157e3c85a74f9841887155095bd163303"
DEFAULT_AVG_NEIGHBORS = 13.735293601560318


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Required environment variable not set: {name}")
    return val


def _ai4s_base() -> Path:
    return Path(_require_env("AI4S_SHARED_DIR")) / "models" / "HydraGNN"


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    hg = config.get("HydraGNN", {})
    base = _ai4s_base()
    repo = Path(os.environ.get("HG_REPO_DIR", base / "code" / "HydraGNN"))
    example = repo / "examples" / "multidataset_hpo_sc26"
    data_dir = Path(os.environ.get("HG_DATA_DIR", base / "weights"))
    output = Path(os.environ.get("HG_OUTPUT_DIR", os.environ.get("OMNIHUB_RESULTS_DIR", ".")))
    return {
        "repo": repo,
        "example": example,
        "data_dir": data_dir,
        "output": output,
        "config_json": example / "gfm_mlip.json",
        "entry": example / "gfm_mlip_all_mpnn.py",
    }


def _ensure_datasets(paths: Dict[str, Path], datasets: str) -> None:
    dataset_dir = paths["example"] / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for ds in datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        target = dataset_dir / f"{ds}-v2.bp"
        source = paths["data_dir"] / f"{ds}-v2.bp"
        if not source.is_dir():
            raise FileNotFoundError(f"Dataset not found: {source}")
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)


def _maybe_inject_profile(config_path: Path, target_epoch: Optional[int]) -> Path:
    """Inject NeuralNetwork.Profile block for epoch-gated kineto (perf-analysis)."""
    if target_epoch is None:
        return config_path
    data = json.loads(config_path.read_text())
    nn = data.setdefault("NeuralNetwork", {})
    nn["Profile"] = {
        "enable": 1,
        "target_epoch": int(target_epoch),
        "wait": 5,
        "warmup": 3,
        "active": 3,
        "repeat": 1,
    }
    out = Path(
        os.environ.get(
            "OMNIHUB_RESULTS_DIR",
            os.environ.get("HG_OUTPUT_DIR", str(config_path.parent)),
        )
    ) / "gfm_mlip_omnihub_profile.json"
    out.write_text(json.dumps(data, indent=2))
    return out


def _patch_adios_avg_neighbors() -> None:
    avg_nn = float(os.environ.get("HYDRAGNN_AVG_NUM_NEIGHBORS", DEFAULT_AVG_NEIGHBORS))
    if avg_nn <= 0:
        return
    import hydragnn.utils.datasets.adiosdataset as adm

    orig_init = adm.AdiosMultiDataset.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.avg_num_neighbors = avg_nn

    adm.AdiosMultiDataset.__init__ = patched_init


def _setup_runtime_env(paths: Dict[str, Path]) -> None:
    scratch = os.environ.get("SCRATCH_LOCAL", "/scratch")
    proc = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))
    job = os.environ.get("SLURM_JOB_ID", "local")
    scratch_rank = Path(scratch) / os.environ.get("USER", "user") / f"hydragnn-{job}" / proc
    scratch_rank.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(scratch_rank)
    os.environ["MIOPEN_DISABLE_CACHE"] = "1"
    os.environ["MIOPEN_USER_DB_PATH"] = str(scratch_rank / "miopen")
    Path(os.environ["MIOPEN_USER_DB_PATH"]).mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["HYDRAGNN_USE_VARIABLE_GRAPH_SIZE"] = "1"
    os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
    os.environ["HYDRAGNN_USE_FSDP"] = "0"
    os.environ["HYDRAGNN_TRACE_LEVEL"] = os.environ.get("HYDRAGNN_TRACE_LEVEL", "1")
    os.environ["HSA_NO_SCRATCH_RECLAIM"] = "1"
    os.environ["HG_EXAMPLE_DIR"] = str(paths["example"])
    os.environ["HG_OUTPUT_DIR"] = str(paths["output"])
    paths["output"].mkdir(parents=True, exist_ok=True)

    pkg = "/opt/hydragnn-pkgs"
    if Path(pkg).is_dir():
        os.environ["PYTHONPATH"] = f"{pkg}:{os.environ.get('PYTHONPATH', '')}"
        ld = f"{pkg}/adios2:/opt/ompi/lib"
        os.environ["LD_LIBRARY_PATH"] = f"{ld}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def _build_argv(paths: Dict[str, Path], hg: Dict[str, Any], config_file: Path) -> list:
    datasets = os.environ.get("HG_DATASETS", hg.get("datasets", "ANI1x,Alexandria"))
    batch_size = int(os.environ.get("HG_BATCH_SIZE", hg.get("batch_size", 200)))
    max_batch = os.environ.get("HYDRAGNN_MAX_NUM_BATCH", hg.get("max_num_batch") or "")
    num_epoch = os.environ.get("HG_NUM_EPOCH", str(hg.get("num_epoch", 1)))
    precision = os.environ.get("HG_PRECISION", hg.get("precision", "fp64"))
    job = os.environ.get("SLURM_JOB_ID", "0")
    nodes = os.environ.get("SLURM_JOB_NUM_NODES", "1")

    argv = [
        str(paths["entry"]),
        f"--inputfile={config_file}",
        "--multi",
        "--everyone",
        f"--multi_model_list={datasets}",
        f"--precision={precision}",
        f"--batch_size={batch_size}",
        f"--num_epoch={num_epoch}",
        "--startfrom=none",
        f"--log=hydragnn-train-{job}-N{nodes}",
    ]
    if max_batch:
        n = int(max_batch) * batch_size
        argv += [
            f"--num_samples={n}",
            "--oversampling",
            f"--oversampling_num_samples={n}",
        ]
    return argv


@omnihub.run.entrypoint
def run(extra_args, config=None):
    """OmniHub entrypoint — train HydraGNN GFM ensemble."""
    hg = (config or {}).get("HydraGNN", {})
    paths = _resolve_paths(config or {})
    datasets = os.environ.get("HG_DATASETS", hg.get("datasets", "ANI1x,Alexandria"))

    if not paths["entry"].is_file():
        raise FileNotFoundError(
            f"HydraGNN entry not found: {paths['entry']}. "
            f"Clone upstream to HG_REPO_DIR (SHA {DEFAULT_HG_SHA})."
        )

    _ensure_datasets(paths, datasets)
    profile_epoch = hg.get("profile_target_epoch")
    if profile_epoch is not None:
        profile_epoch = int(profile_epoch)
    elif os.environ.get("PROFILE_TARGET_EPOCH"):
        profile_epoch = int(os.environ["PROFILE_TARGET_EPOCH"])

    config_file = _maybe_inject_profile(paths["config_json"], profile_epoch)
    _setup_runtime_env(paths)

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    if os.environ.get("PROFILE_RANK0_ONLY", "1") == "1" and rank != 0:
        os.environ["HYDRAGNN_DISABLE_PROFILE"] = "1"

    @omnihub.tools.profile()
    def _train():
        _patch_adios_avg_neighbors()
        os.chdir(paths["output"])
        sys.argv = _build_argv(paths, hg, config_file)
        runpy.run_path(str(paths["entry"]), run_name="__main__")

    _train()


if __name__ == "__main__":
    run([], {})
