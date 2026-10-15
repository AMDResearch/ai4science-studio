"""Site-specific settings for the studio backend, read from the environment.

Nothing cluster-specific (shared directories, SLURM partition or account) is
hardcoded in the backend. Values come from environment variables, optionally
loaded from ``studio/.env`` at import time (see ``studio/.env.example``).
Variables already present in the environment always win over the file.

Demo (replay) mode needs none of these. Live SLURM runs need at least
``AI4S_SHARED_DIR`` and a partition (``AI4S_SLURM_PARTITION`` or the partition
chosen in the UI); a missing value raises ``SiteConfigError`` with a message
that names the variable to set.
"""
from __future__ import annotations

import os
from pathlib import Path

STUDIO_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = Path(os.environ.get("STUDIO_ENV_FILE") or (STUDIO_DIR / ".env"))

# Container image shared by the live model recipes (relative to AI4S_SHARED_DIR).
_DEFAULT_SIF_NAME = "images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif"


class SiteConfigError(RuntimeError):
    """A required site setting is missing."""


def _load_env_file(path: Path) -> None:
    """Minimal KEY=VALUE loader (no python-dotenv dependency).

    Blank lines and ``#`` comments are skipped, an optional ``export`` prefix is
    accepted, surrounding quotes are stripped, and ``${VAR}`` references are
    expanded against the variables defined so far.
    """
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = os.path.expandvars(value)


_load_env_file(ENV_FILE)


def _get(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _require(name: str, purpose: str) -> str:
    value = _get(name)
    if not value:
        raise SiteConfigError(
            f"{name} is not set. {purpose} Set it in the environment or in "
            f"{ENV_FILE} (see studio/.env.example)."
        )
    return value


def shared_dir() -> str:
    """Site shared directory holding images, overlays, weights, and run output."""
    return _require(
        "AI4S_SHARED_DIR",
        "Live SLURM runs need the site shared directory (container images, overlays, weights).",
    ).rstrip("/")


def default_partition() -> str:
    """Default SLURM partition for live runs, or '' when not configured."""
    return _get("AI4S_SLURM_PARTITION")


def slurm_partition(requested: str | None = None) -> str:
    """The partition to submit to: the UI choice, else AI4S_SLURM_PARTITION."""
    value = (requested or "").strip() or default_partition()
    if not value:
        raise SiteConfigError(
            "No SLURM partition selected and AI4S_SLURM_PARTITION is not set. "
            f"Pick a partition in the Run step or set it in {ENV_FILE} (see studio/.env.example)."
        )
    return value


def slurm_account() -> str:
    """SLURM account for live runs, or '' to let SLURM use the user's default."""
    return _get("AI4S_SLURM_ACCOUNT")


def sif_path() -> str:
    return _get("AI4S_SIF") or f"{shared_dir()}/{_DEFAULT_SIF_NAME}"


def perf_tools_dir() -> str:
    """Omnistat venv, rocprofiler extension, and VictoriaMetrics binary."""
    return _get("PERF_TOOLS_DIR") or f"{shared_dir()}/perf-tools"


def hydragnn_base() -> str:
    return _get("HG_BASE") or f"{shared_dir()}/models/HydraGNN"


def hydragnn_data_dir() -> str:
    """Directory holding the Alexandria ADIOS dataset for HydraGNN."""
    return _get("HG_DATA_DIR") or f"{hydragnn_base()}/weights"


def orbit2_base() -> str:
    return _get("ORBIT2_BASE") or f"{shared_dir()}/models/ORBIT-2"


def orbit2_root() -> str:
    """ORBIT-2 upstream code checkout."""
    return _get("ORBIT2_ROOT") or f"{orbit2_base()}/code/ORBIT-2"


def orbit2_hf_cache() -> str:
    """Directory holding the pretrained ORBIT-2 checkpoints downloaded from Hugging Face."""
    return _get("ORBIT2_HF_CACHE") or os.path.expanduser("~/.cache/huggingface/orbit2")


def orbit2_sr_dir() -> str:
    """Working directory of the ORBIT-2 DC downscaling study (checkpoints, urban grid)."""
    return _get("ORBIT2_SR_DIR") or f"{shared_dir()}/orbit2_sr"


def gpmolformer_base() -> str:
    return _get("GPMOL_BASE") or f"{shared_dir()}/models/GP-MoLFormer"
