"""Job management — SLURM (live) and synthetic (demo) runs."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import slurm
import provenance

RUNS_DIR = Path(__file__).parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# In-memory state registry: id -> dict
_jobs: dict[str, dict] = {}
_queues: dict[str, asyncio.Queue] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


import re as _re

# Noisy lines from OpenMPI/PMIx/SLURM that clutter the run log without signal.
# Dropped at the single choke point so output.log itself stays clean.
_LOG_DENY = [
    _re.compile(p, _re.IGNORECASE) for p in (
        r"Sorry!\s+You were supposed to get help",
        r"Couldn't open the help file.*\.txt",
        r"help-btl-vader\.txt", r"btl_vader",
        r"But I couldn't open the help file",
        r"No such file or directory.*help",
        r"PMIx?\b.*(WARNING|not found|unable)",
        r"ompi_mpi_init|MPI_Init.*warn",
        r"^\s*-{10,}\s*$",  # bare separator rules echoed by MPI help blocks
    )
]


def _emit(run_id: str, line: str):
    """Append to log file and push to SSE queue if any listener.

    Drops known-noisy MPI/PMIx help-file lines (see _LOG_DENY) so both the on-disk
    log and the live stream stay signal-only.
    """
    if any(rx.search(line) for rx in _LOG_DENY):
        return
    log = _run_dir(run_id) / "output.log"
    with open(log, "a") as f:
        f.write(line + "\n")
    q = _queues.get(run_id)
    if q:
        q.put_nowait(line)


def _update_state(run_id: str, state: str, slurm_job_id: str | None = None):
    job = _jobs.get(run_id, {})
    job["state"] = state
    job["updated_at"] = _now()
    if slurm_job_id is not None:
        job["slurm_job_id"] = slurm_job_id
    _jobs[run_id] = job
    provenance.upsert_run(job)


# ── Synthetic (demo) runner ───────────────────────────────────────────────────

def _run_synthetic(run_id: str, slug: str, domain: str, task: str, prompt: str, params: dict):
    """Runs in a thread. Calls the synthetic generator and streams fake log lines."""
    import synthetic as syn
    # These demos replay REAL pre-computed results (not synthetic): the ORBIT-2
    # story and every HydraGNN task (inference replays real predictions on
    # held-out Alexandria DFT; training replays real 1-GPU/8-GPU run logs).
    _is_replay = (slug == "ORBIT-2" and task == "story") or slug == "HydraGNN"
    if _is_replay:
        _src = ("real PRISM data, real finetunes on MI355X" if slug == "ORBIT-2"
                else "real HydraGNN runs on Alexandria DFT, pre-computed on MI355X")
        _emit(run_id, f"[studio] Demo mode — replaying real pre-computed results for {slug} (not synthetic)...")
        _emit(run_id, f"[studio] Task: {task} | Domain: {domain}")
        _emit(run_id, f"[studio] Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        _emit(run_id, f"[studio] Loading measured results ({_src})...")
    else:
        _emit(run_id, f"[studio] Demo mode — generating synthetic output for {slug}...")
        _emit(run_id, f"[studio] Task: {task} | Domain: {domain}")
        _emit(run_id, f"[studio] Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        _emit(run_id, "[studio] Initializing synthetic data generator...")
    time.sleep(0.5)
    _update_state(run_id, "running")

    out_dir = _run_dir(run_id) / "output"
    out_dir.mkdir(exist_ok=True)

    try:
        result = syn.generate(slug, domain, task, prompt, params, out_dir, _emit, run_id)
        _emit(run_id, f"[studio] Done. Output in: {out_dir}")
        _emit(run_id, f"[studio] Result summary: {json.dumps(result)}")
        job = _jobs.get(run_id, {})
        job["output_dir"] = str(out_dir)
        job["result"] = result
        _jobs[run_id] = job
        _update_state(run_id, "completed")
    except Exception as e:
        _emit(run_id, f"[studio] ERROR: {e}")
        _update_state(run_id, "failed")
    finally:
        q = _queues.get(run_id)
        if q:
            q.put_nowait(None)  # sentinel


# ── SLURM (live) runner ───────────────────────────────────────────────────────

_AI4S_SHARED_DIR = "/shared/spannala"
_BACKEND_DIR     = Path(__file__).resolve().parent
_SLURM_ACCOUNT   = "vultr_lux"
_SIF_PATH        = f"{_AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif"

# Per-model overlay paths (None = no overlay needed)
_OVERLAYS: dict[str, str | None] = {
    "HydraGNN":    f"{_AI4S_SHARED_DIR}/models/HydraGNN/overlays/hydragnn-overlay.img",
    "ORBIT-2":     f"{_AI4S_SHARED_DIR}/models/ORBIT-2/overlays/orbit2-overlay.img",
    "StormCast":   f"{_AI4S_SHARED_DIR}/models/StormCast/overlays/stormcast-overlay.img",
    "GP-MoLFormer": None,
}


def _is_dc_run(slug: str, task: str, params: dict) -> bool:
    """A real DC physics-residual-head run: the ORBIT-2 'story' task OR any ORBIT-2
    prompt carrying a dc_event (the DC heatwave inference prompts). These all run the
    orbit2_ood_dc.py harness, never the generic synthetic downscaling path."""
    return slug == "ORBIT-2" and (task == "story" or "dc_event" in params)


def _build_slurm_script(run_id: str, slug: str, domain: str, task: str,
                         prompt: str, params: dict, partition: str) -> Path:
    """Build a minimal sbatch wrapper that echoes params and calls the model's sbatch script."""
    import shlex
    from registry import REPO_ROOT
    q = shlex.quote  # POSIX-safe shell quoting for any interpolated value
    model_dir = REPO_ROOT / domain / "models" / slug
    log_path = str(_run_dir(run_id) / "slurm.log")

    # Pick the most specific sbatch script (prefer task-matching name).
    if slug == "HydraGNN" and task == "train":
        # Live 8-GPU DDP training wrapped with Omnistat telemetry. Self-contained in
        # studio/telemetry/ (writes only to a per-run dir; never touches the demo
        # checkpoint). Not under model_dir/examples, so reference it directly.
        _tele = _BACKEND_DIR.parent / "telemetry" / "sbatch_train_telemetry_amd.sh"
        sbatch_candidates = [_tele] if _tele.exists() else []
    elif slug == "HydraGNN":
        # Use our trained-model inference script; DDP scripts are for the training workflow.
        sbatch_candidates = list(model_dir.glob("examples/sbatch_studio_infer_amd.sh"))
    elif slug == "GP-MoLFormer" and task == "finetune":
        # Pair-tuning: prefer the dedicated pairtune sbatch if it exists.
        sbatch_candidates = (
            list(model_dir.glob("examples/sbatch*pairtune*amd*.sh")) +
            list(model_dir.glob("examples/sbatch*finetune*amd*.sh"))
        )
    else:
        sbatch_candidates = (
            list(model_dir.glob(f"examples/sbatch*{task}*amd*.sh")) +
            list(model_dir.glob("examples/sbatch*infer*amd*.sh")) +
            list(model_dir.glob("examples/sbatch*amd*.sh"))
        )
    upstream_script = sbatch_candidates[0] if sbatch_candidates else None

    overlay = _OVERLAYS.get(slug)

    # Resource header. Default = single-GPU inference/story path. The 8-GPU HydraGNN
    # training case (live telemetry) needs all 8 GPUs, one task per GPU, and a longer
    # wall time. Parameterized here rather than string-literal so future multi-GPU
    # cases don't each need a new special case.
    _is_hydragnn_train = (slug == "HydraGNN" and task == "train")
    if _is_hydragnn_train:
        _gpus, _ntasks, _time = 8, 8, "01:00:00"
    else:
        # One task per node so models that fan out via an inner `srun --mpi=pmix`
        # (e.g. ORBIT-2) get a bindable CPU mask. Do NOT pin --cpus-per-task: a
        # constrained mask makes the nested srun fail "Unable to satisfy cpu bind".
        _gpus, _ntasks, _time = 1, 1, "00:30:00"

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=studio-{slug[:10]}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --account={_SLURM_ACCOUNT}",
        "#SBATCH --nodes=1",
        f"#SBATCH --gres=gpu:{_gpus}",
        f"#SBATCH --ntasks-per-node={_ntasks}",
        f"#SBATCH --time={_time}",
        f"#SBATCH --output={log_path}",
        f"#SBATCH --error={log_path}",
        "",
        f"echo {q(f'[studio] Run ID: {run_id}')}",
        f"echo {q(f'[studio] Model: {slug} | Task: {task}')}",
        f"echo {q(f'[studio] Partition: {partition}')}",
        f"echo {q(f'[studio] Prompt: {prompt[:100]}')}",
        "",
        # Core env vars every upstream script needs
        f"export AI4S_SHARED_DIR={_AI4S_SHARED_DIR!r}",
        f"export HG_SIF={_SIF_PATH!r}",
        f"export ORBIT2_SIF={_SIF_PATH!r}",
        f"export GPMOL_SIF={_SIF_PATH!r}",
        f"export SC_SIF={_SIF_PATH!r}",
    ]

    if overlay:
        script_lines += [
            f"export HG_OVERLAY={overlay!r}",
            f"export ORBIT2_OVERLAY={overlay!r}",
            f"export SC_OVERLAY={overlay!r}",
        ]

    # Point dataset dirs at aaji's pre-staged data where available
    script_lines += [
        f"export HG_DATA_DIR=/shared/aaji/models/HydraGNN/weights",
        "",
    ]

    # ORBIT-2 runs self-contained synthetic downscaling inference: the upstream
    # code clone lives in aaji's tree, and synthetic mode auto-generates data +
    # auto-downloads the checkpoint from HF. Single-GPU config avoids the 16-way
    # FSDP default. UI params below can override any of these.
    if _is_dc_run(slug, task, params):
        # Real 4-act robustness/finetuning story: PRISM-trained ORBIT-2 applied to
        # INDEPENDENT Open-Meteo DC heatwave data (true out-of-distribution test).
        # Static channels come from the PRISM DC crop; dynamic (tmax/tmin/precip)
        # from the baked real Open-Meteo fields. DC event selectable via dc_event.
        # Runs for BOTH the 'story' task and the DC-event 'inference' prompts.
        _O2 = f"{_AI4S_SHARED_DIR}/orbit2_sr"
        _OOD = str(_BACKEND_DIR / "assets" / "dc_ood_fields.json")
        _dc_event = str(params.get("dc_event", "july16_2024"))
        script_lines += [
            f"export OUT_JSON={str(_run_dir(run_id) / 'orbit2_story.json')}",
            "_U=(); while IFS= read -r v; do _U+=(-u \"$v\"); done "
            "< <(env | grep -oE '^(PMIX_|PMI_|OMPI_)[A-Za-z0-9_]+')",
            "srun --mpi=pmix --ntasks=1 --gpus-per-node=1 --cpu-bind=none "
            f"env \"${{_U[@]}}\" apptainer exec --rocm --overlay {overlay}:ro "
            "--bind /shared/aaji/models/ORBIT-2:/shared/aaji/models/ORBIT-2:ro "
            "--bind /home/spannala/.cache/huggingface/orbit2:/home/spannala/.cache/huggingface/orbit2:ro "
            f"--bind {_AI4S_SHARED_DIR}:{_AI4S_SHARED_DIR} "
            f"--bind {_BACKEND_DIR}:{_BACKEND_DIR}:ro "
            f"--bind {_run_dir(run_id)}:{_run_dir(run_id)} "
            "--env PYTHONPATH=/opt/orbit2-pkgs:/shared/aaji/models/ORBIT-2/code/ORBIT-2/src "
            "--env PYTHONNOUSERSITE=1 --env HSA_NO_SCRATCH_RECLAIM=1 --env MIOPEN_DISABLE_CACHE=1 "
            f"--env OOD_FIELDS={_OOD} --env DC_EVENT={_dc_event} "
            f"--env URBAN_CONUS={_O2}/urban_conus.npz "
            f"--env RH_CKPT={_O2}/orbit2_residual_head.pk "
            "--env WATER_L_CELLS=3.0 "
            f"--env OUT_JSON={str(_run_dir(run_id) / 'orbit2_story.json')} "
            f"{_SIF_PATH} bash -lc "
            f"'source /opt/venv/bin/activate 2>/dev/null; python3 {_O2}/orbit2_ood_dc.py'",
            "",
        ]
    elif slug == "ORBIT-2":
        script_lines += [
            f"export ORBIT2_ROOT=/shared/aaji/models/ORBIT-2/code/ORBIT-2",
            f"export ORBIT2_USE_SYNTHETIC=1",
            f"export ORBIT2_CONFIG=interm_8m_synthetic_1gpu.yaml",
            f"export ORBIT2_OUTPUT_DIR={_AI4S_SHARED_DIR}/models/ORBIT-2/outputs/{run_id}",
            "",
        ]

    # HydraGNN: predict energy with OUR trained PNA model on a real held-out material.
    _HG_WORK = f"{_AI4S_SHARED_DIR}/models/HydraGNN/train_work"
    _PERF_TOOLS_DIR = f"{_AI4S_SHARED_DIR}/perf-tools"
    if slug == "HydraGNN" and task == "train":
        # Live 8-GPU DDP training + Omnistat telemetry. The telemetry sbatch reads
        # these; it writes ONLY to a per-run perf-runs/<jobid> dir (HG_OUTPUT_DIR is
        # left as the sbatch default, keyed by SLURM_JOB_ID) so it never overwrites
        # the production checkpoint the inference demos load. STUDIO_TELEMETRY_DIR is
        # recorded so _harvest_telemetry can find the run's manifest.json afterward.
        _epochs = int(params.get("epochs", 200))
        _prec = str(params.get("precision", "fp32")).lower()
        if _prec not in ("fp32", "fp64"):
            _prec = "fp32"
        # Sampling resolution (s) and kernel-trace toggle (UI-selectable).
        try:
            _interval = float(params.get("omnistat_interval", 0.5))
        except (TypeError, ValueError):
            _interval = 0.5
        _interval = min(max(_interval, 0.1), 10.0)
        _ktrace = 1 if str(params.get("kernel_trace", 0)) in ("1", "true", "True") else 0
        script_lines += [
            f"export PERF_TOOLS_DIR={_PERF_TOOLS_DIR!r}",
            f"export OMNISTAT_VENV={_PERF_TOOLS_DIR}/omnistat-venv",
            f"export HG_DATA_DIR={_AI4S_SHARED_DIR}/models/HydraGNN/weights",
            f"export HG_NUM_EPOCH={_epochs}",
            f"export HG_PRECISION={_prec}",
            f"export OMNISTAT_USERMODE_INTERVAL={_interval}",
            f"export OMNISTAT_KERNEL_TRACE={_ktrace}",
            "",
        ]
    elif slug == "HydraGNN":
        # Pick checkpoint by variant. Both models now use the SAME stored-PBC-edge
        # pipeline and identical config, so the 1-GPU vs 8-GPU comparison isolates
        # data scale + GPU count (the honest scaling story): 8gpu = v3 (600k
        # structures, MAE 0.24 / corr 0.89); 1gpu = hg_model_1gpu_pbc (40k, MAE
        # 0.29 / corr 0.82). Both need HG_USE_PBC_EDGES=1.
        variant = str(params.get("model_variant", "8gpu")).lower()
        ckpt = "hg_model_ddp_v3.pk" if variant == "8gpu" else "hg_model_1gpu_pbc.pk"
        # The 4 suggested prompts each carry a curated slot (struct_index 0-3), which
        # maps to a fixed valset index below. Same order as CURATED_INDICES and the
        # demo predictions JSON: 0=FeS2, 1=FeNi3, 2=NaFeO2, 3=Fe2H6.
        _CURATED = [2816, 3372, 29321, 1349]
        _has_slot = ("struct_index" in params) or ("STRUCT_INDEX" in params)
        _slot = params.get("struct_index", params.get("STRUCT_INDEX", 4))
        pl = prompt.lower()
        if _has_slot and isinstance(_slot, (int, float)) and 0 <= int(_slot) < len(_CURATED):
            # User picked one of the 4 curated materials: pin its exact held-out
            # structure and suppress element-based scanning/selection entirely, so
            # they always get back the material they clicked (not a lookalike).
            struct_index = _CURATED[int(_slot)]
            req = ""
        else:
            # Custom prompt: derive an element filter from the prompt text so the
            # shown structure matches what was asked (e.g. "iron-carbon" -> Fe+C).
            struct_index = int(_slot)
            _ELEM = {"iron": 26, "carbon": 6, "oxygen": 8, "lithium": 3, "li": 3,
                     "cobalt": 27, "nickel": 28, "manganese": 25, "titanium": 22, "silicon": 14}
            req = params.get("require_elements")
            if not req:
                hits = sorted({z for name, z in _ELEM.items() if name in pl})
                req = ",".join(str(z) for z in hits) if hits else ""
        script_lines += [
            f"export HG_INFER_REPO={_AI4S_SHARED_DIR}/models/HydraGNN/outputs/HydraGNN-infer",
            f"export HG_MODEL_PATH={_HG_WORK}/results/{ckpt}",
            f"export HG_MODEL_VARIANT={variant!r}",
            f"export HG_USE_PBC_EDGES=1",
            f"export STRUCT_INDEX={int(struct_index)}",
            f"export REQUIRE_ELEMENTS={q(str(req))}",
            # Curated small demo materials (FeS2, FeNi3, NaFeO2, Fe2H6) + their
            # name/application labels. studio_infer.py falls back to these when no
            # element filter is requested, and enforces a size cap when one is.
            f"export CURATED_INDICES='2816,3372,29321,1349'",
            f"export MATERIAL_LABELS_FILE={_HG_WORK}/inference/demo_material_labels.json",
            f"export HG_STUDIO_INFER={_HG_WORK}/inference/studio_infer.py",
            f"export STUDIO_RESULT_OUT={str(_run_dir(run_id) / 'hg_result.json')}",
            "",
        ]

    # GP-MoLFormer: pass pairtune params and result output path for finetune task.
    _GPMOL_WORK = f"{_AI4S_SHARED_DIR}/models/GP-MoLFormer"
    if slug == "GP-MoLFormer" and task == "finetune":
        prop = str(params.get("pairtune_prop", "qed")).lower()
        epochs = int(params.get("pairtune_epochs", 10))
        script_lines += [
            f"export GPMOL_WORK_DIR={_GPMOL_WORK!r}",
            f"export PAIRTUNE_PROP={prop!r}",
            f"export PAIRTUNE_EPOCHS={epochs}",
            f"export STUDIO_RESULT_OUT={str(_run_dir(run_id) / 'gpmol_finetune_result.json')!r}",
            "",
        ]

    # Extra params from UI (skip ones already handled above)
    _handled = {"model_variant", "struct_index", "pairtune_prop", "pairtune_epochs",
                "dc_event", "model_variant", "STRUCT_INDEX", "epochs", "precision",
                "omnistat_interval", "kernel_trace"}
    for k, v in params.items():
        if k.lower() not in _handled:
            script_lines.append(f"export {k.upper()}={q(str(v))}")
    script_lines.append("")

    # The ORBIT-2 DC runs (story + DC-event inference) run their own inline srun
    # harness above — do not also delegate to the upstream (synthetic) sbatch.
    _story_run = _is_dc_run(slug, task, params)

    if upstream_script and not _story_run:
        # Upstream scripts resolve their own dir via `scontrol show job`, which returns
        # THIS generated job.sh (not the examples dir) when we bash-call them. Pass the
        # real examples dir explicitly so their /examples bind-mount points at the scripts.
        script_lines.append(f"export STUDIO_EXAMPLES_DIR={str(upstream_script.parent)!r}")
        script_lines.append(f"echo '[studio] Delegating to upstream script: {upstream_script.name}'")
        script_lines.append(f"bash {upstream_script} 2>&1")
    elif _story_run:
        script_lines.append("echo '[studio] ORBIT-2 story harness complete'")
    else:
        script_lines.append(f"echo '[studio] No upstream sbatch script found for {slug}'")
        script_lines.append("echo '[studio] Model output would appear here in a real run'")
        script_lines.append("sleep 10")
        script_lines.append("echo '[studio] SLURM job complete'")

    script_path = _run_dir(run_id) / "job.sh"
    script_path.write_text("\n".join(script_lines))
    script_path.chmod(0o755)
    return script_path


def _tail_slurm_log(run_id: str, slurm_log: Path):
    """Tail the SLURM output log and emit new lines to the run stream."""
    if not slurm_log.exists():
        return
    try:
        with open(slurm_log) as f:
            for line in f:
                line = line.rstrip()
                if line:
                    _emit(run_id, line)
    except Exception:
        pass


def _harvest_telemetry(run_id: str) -> dict | None:
    """Build the training_telemetry result for a completed live 8-GPU HydraGNN run.

    Locates the per-run perf dir (train_work/perf-runs/<slurm_job_id>), queries its
    Omnistat VictoriaMetrics DB via telemetry.harvest(), and folds in the final
    validation metrics (corr/mae/r2) the training wrote to validation.json.
    """
    import telemetry
    job = _jobs.get(run_id, {})
    slurm_id = job.get("slurm_job_id")
    if not slurm_id:
        return None
    perf_dir = Path(f"{_AI4S_SHARED_DIR}/models/HydraGNN/train_work/perf-runs/{slurm_id}")
    manifest = perf_dir / "manifest.json"
    if not manifest.exists():
        _emit(run_id, f"[studio] telemetry manifest not found: {manifest}")
        return None

    tel = telemetry.harvest(manifest)
    result: dict = {
        "type": "training_telemetry",
        "slug": "HydraGNN",
        "model": "HydraGNN (Predictive GFM 2024)",
    }
    if tel:
        result.update({
            "n_gpus": tel.get("n_gpus", 8),
            "epochs": tel.get("epochs"),
            "runtime_s": tel.get("runtime_s"),
            "telemetry": {k: tel[k] for k in ("peaks", "means", "units", "series") if k in tel},
        })
    # Fold in final accuracy from the training's own validation.json (verbatim).
    val_json = perf_dir / "validation.json"
    if val_json.exists():
        try:
            v = json.loads(val_json.read_text())
            m = v.get("metrics", v)  # metrics are nested under "metrics"
            result["metrics"] = {k: m.get(k) for k in ("corr", "mae", "r2") if k in m}
        except Exception:
            pass

    # Parse per-epoch loss curve so the Results tab shows convergence. The studio
    # backend delegates to the telemetry sbatch via `bash sbatch_...`, so the inner
    # `#SBATCH --output` is inert — training stdout lands in THIS run's slurm.log /
    # output.log (not train_work/logs/hg_tele8_<jobid>.log). Read the run-dir logs.
    import re
    pat = re.compile(
        r"Epoch:\s*(\d+),\s*Train Loss:\s*([\d.eE+-]+),\s*"
        r"Val Loss:\s*([\d.eE+-]+),\s*Test Loss:\s*([\d.eE+-]+)")
    _candidates = [
        _run_dir(run_id) / "slurm.log",
        _run_dir(run_id) / "output.log",
        Path(f"{_AI4S_SHARED_DIR}/models/HydraGNN/train_work/logs/hg_tele8_{slurm_id}.log"),
    ]
    for log in _candidates:
        if not log.exists():
            continue
        try:
            epochs = []
            for line in log.read_text().splitlines():
                mm = pat.search(line)
                if mm:
                    epochs.append({"ep": int(mm.group(1)), "train": float(mm.group(2)),
                                   "val": float(mm.group(3)), "test": float(mm.group(4))})
            if epochs:
                result["loss_curve"] = epochs
                break
        except Exception:
            pass
    job["output_dir"] = str(perf_dir)
    _jobs[run_id] = job
    return result if (result.get("telemetry") or result.get("metrics")) else None


def _harvest_results(run_id: str):
    """Parse real model output from a completed LIVE run into job['result'] + output_dir.

    Values are extracted verbatim from the model's own stdout/output files — never
    synthesized. Only fields the model actually produced are populated.
    """
    import re
    job = _jobs.get(run_id, {})
    slug = job.get("slug", "")
    log = _run_dir(run_id) / "output.log"
    text = log.read_text() if log.exists() else ""

    if slug == "ORBIT-2" and (_run_dir(run_id) / "orbit2_story.json").exists():
        # Any DC run (story or DC-event inference): the harness wrote a complete
        # 4-act story JSON — read it verbatim. Generic synthetic runs write no such
        # file and fall through to the regex-harvest path below.
        try:
            job["result"] = json.loads((_run_dir(run_id) / "orbit2_story.json").read_text())
            _jobs[run_id] = job
            _emit(run_id, "[studio] ORBIT-2 story results harvested.")
        except Exception as e:
            _emit(run_id, f"[studio] Could not parse ORBIT-2 story: {e}")
        return

    if slug == "ORBIT-2":
        out_dir = f"{_AI4S_SHARED_DIR}/models/ORBIT-2/outputs/{run_id}"
        result: dict = {"type": "downscaling", "model": "ORBIT-2"}
        # "Goodness of fit: PSNR 14.430530, SSIM 0.029594"
        m = re.search(r"PSNR\s+([-\d.]+),\s*SSIM\s+([-\d.]+)", text)
        if m:
            result["psnr"] = round(float(m.group(1)), 4)
            result["ssim"] = round(float(m.group(2)), 4)
        # "img.shape (20, 40), min 3.46..., max 9.49..."
        mi = re.search(r"img\.shape\s*\(([\d, ]+)\),\s*min\s+([-\d.]+),\s*max\s+([-\d.]+)", text)
        if mi:
            result["input_shape"] = [int(x) for x in mi.group(1).split(",")]
            result["input_min"] = round(float(mi.group(2)), 4)
            result["input_max"] = round(float(mi.group(3)), 4)
        # "ppred.shape (80, 160), min 1.13..., max 11.24..."
        mp = re.search(r"ppred\.shape\s*\(([\d, ]+)\),\s*min\s+([-\d.]+),\s*max\s+([-\d.]+)", text)
        if mp:
            result["prediction_shape"] = [int(x) for x in mp.group(1).split(",")]
            result["prediction_min"] = round(float(mp.group(2)), 4)
            result["prediction_max"] = round(float(mp.group(3)), 4)
        result["variable"] = "total_precipitation_24hr"
        if result.get("input_shape") and result.get("prediction_shape"):
            iy, ix = result["input_shape"]
            py, px = result["prediction_shape"]
            if iy and ix:
                result["upscale_factor"] = f"{px // ix}x"
        job["result"] = result
        job["output_dir"] = out_dir
        _jobs[run_id] = job
        _emit(run_id, f"[studio] Results harvested: {json.dumps(result)}")

    elif slug == "HydraGNN" and job.get("task") == "train":
        # Live 8-GPU training run: harvest Omnistat GPU telemetry from the per-run
        # VictoriaMetrics DB. The sbatch keyed HG_OUTPUT_DIR by SLURM_JOB_ID, so the
        # manifest lives at train_work/perf-runs/<slurm_job_id>/manifest.json.
        _res = _harvest_telemetry(run_id)
        if _res:
            job["result"] = _res
            _jobs[run_id] = job
            _emit(run_id, "[studio] Telemetry harvested from Omnistat.")
        else:
            _emit(run_id, "[studio] No telemetry harvested (run may lack Omnistat data).")

    elif slug == "HydraGNN":
        # studio_infer.py writes a complete result JSON; read it verbatim.
        rjson = _run_dir(run_id) / "hg_result.json"
        if rjson.exists():
            try:
                job["result"] = json.loads(rjson.read_text())
                _jobs[run_id] = job
                _emit(run_id, f"[studio] Results harvested: {rjson.read_text()}")
            except Exception as e:
                _emit(run_id, f"[studio] Could not parse HydraGNN result: {e}")

    elif slug == "GP-MoLFormer":
        # sbatch_pairtune_amd.sh writes a result JSON for finetune task;
        # inference task parses generated_molecules.csv from workspace.
        rjson = _run_dir(run_id) / "gpmol_finetune_result.json"
        if rjson.exists():
            try:
                job["result"] = json.loads(rjson.read_text())
                _jobs[run_id] = job
                _emit(run_id, f"[studio] GP-MoLFormer finetune results harvested.")
            except Exception as e:
                _emit(run_id, f"[studio] Could not parse GP-MoLFormer finetune result: {e}")
        else:
            # Inference: parse generated_molecules.csv from GPMOL_WORK_DIR
            work = f"{_AI4S_SHARED_DIR}/models/GP-MoLFormer"
            csv_path = Path(work) / "generated.csv"
            if csv_path.exists():
                try:
                    import csv as csv_mod
                    molecules = []
                    with open(csv_path) as f:
                        for row in csv_mod.reader(f):
                            s = row[0].strip() if row else ""
                            if s and s.lower() != "smiles":
                                molecules.append({"smiles": s})
                    job["result"] = {
                        "type": "molecule_generation",
                        "model": "GP-MoLFormer",
                        "n_generated": len(molecules),
                        "n_valid": len(molecules),
                        "lipinski_pass_rate": None,
                        "molecules": molecules[:20],
                    }
                    _jobs[run_id] = job
                    _emit(run_id, f"[studio] {len(molecules)} molecules harvested.")
                except Exception as e:
                    _emit(run_id, f"[studio] Could not parse GP-MoLFormer output: {e}")


def _poll_slurm(run_id: str, slurm_job_id: str):
    """Poll SLURM job state until terminal; streams progress and job output to log."""
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
    slurm_log = _run_dir(run_id) / "slurm.log"
    last_state = ""
    last_log_size = 0

    for _ in range(360):  # max 30 min at 5s intervals
        info = slurm.job_state(slurm_job_id)
        state = info.get("state", "UNKNOWN")
        if state != last_state:
            _emit(run_id, f"[slurm] Job {slurm_job_id} state: {state}")
            last_state = state

        # Stream any new lines from the SLURM output log
        if slurm_log.exists():
            current_size = slurm_log.stat().st_size
            if current_size > last_log_size:
                try:
                    with open(slurm_log) as f:
                        f.seek(last_log_size)
                        for line in f:
                            line = line.rstrip()
                            if line:
                                _emit(run_id, line)
                    last_log_size = current_size
                except Exception:
                    pass

        if state in terminal:
            # Final drain of log
            if slurm_log.exists():
                try:
                    with open(slurm_log) as f:
                        f.seek(last_log_size)
                        for line in f:
                            line = line.rstrip()
                            if line:
                                _emit(run_id, line)
                except Exception:
                    pass
            final = "completed" if state == "COMPLETED" else "failed"
            if final == "completed":
                _harvest_results(run_id)
            _update_state(run_id, final)
            q = _queues.get(run_id)
            if q:
                q.put_nowait(None)
            return
        time.sleep(5)
    _emit(run_id, "[slurm] Polling timed out")
    _update_state(run_id, "failed")
    q = _queues.get(run_id)
    if q:
        q.put_nowait(None)


# ── Public API ────────────────────────────────────────────────────────────────

def launch(slug: str, domain: str, task: str, mode: str, prompt: str,
           params: dict, partition: str = "lux") -> str:
    run_id = str(uuid.uuid4())
    now = _now()
    job = {
        "id": run_id,
        "slug": slug,
        "domain": domain,
        "task": task,
        "mode": mode,
        "prompt": prompt,
        "params": params,
        "partition": partition,
        "state": "pending",
        "slurm_job_id": None,
        "output_dir": None,
        "result": None,
        "created_at": now,
        "updated_at": now,
    }
    _jobs[run_id] = job
    _queues[run_id] = asyncio.Queue()
    provenance.upsert_run(job)

    if mode == "demo":
        import threading
        t = threading.Thread(target=_run_synthetic, args=(run_id, slug, domain, task, prompt, params), daemon=True)
        t.start()
    else:
        # Live SLURM
        script = _build_slurm_script(run_id, slug, domain, task, prompt, params, partition)
        ok, slurm_id = slurm.submit(str(script))
        if ok:
            _emit(run_id, f"[studio] SLURM job submitted: {slurm_id}")
            _update_state(run_id, "queued", slurm_job_id=slurm_id)
            import threading
            t = threading.Thread(target=_poll_slurm, args=(run_id, slurm_id), daemon=True)
            t.start()
        else:
            _emit(run_id, f"[studio] SLURM submission failed: {slurm_id}")
            _update_state(run_id, "failed")
            q = _queues.get(run_id)
            if q:
                q.put_nowait(None)

    return run_id


def get_job(run_id: str) -> dict | None:
    return _jobs.get(run_id) or provenance.get_run(run_id)


def live_telemetry(run_id: str, keys: list[str] | None) -> dict:
    """Query the running job's Omnistat VM (compute node :9090) for live telemetry.

    Returns {status, telemetry?}: 'pending' (no SLURM job / node yet), 'starting'
    (VM up but no samples pushed — omnistat's first push is ≥1 min in), or 'live'.
    """
    import telemetry
    from datetime import datetime
    job = _jobs.get(run_id) or provenance.get_run(run_id) or {}
    # If the run already completed and harvested, serve the final result telemetry.
    if job.get("state") == "completed" and (job.get("result") or {}).get("telemetry"):
        return {"status": "final", "telemetry": job["result"]["telemetry"]}
    slurm_id = job.get("slurm_job_id")
    if not slurm_id:
        return {"status": "pending"}
    info = slurm.job_state(str(slurm_id))
    node = info.get("node")
    if info.get("state", "").upper() in ("PENDING", "") or not node:
        return {"status": "pending"}
    # Derive the job's start epoch from created_at so the live window grows from t0.
    start_epoch = 0
    try:
        start_epoch = int(datetime.fromisoformat(job["created_at"]).timestamp())
    except Exception:
        pass
    tel = telemetry.harvest_live(node, start_epoch, keys=keys)
    if tel is None:
        return {"status": "starting", "node": node}
    return {"status": "live", "node": node, "telemetry": tel}


def list_jobs(limit: int = 50) -> list[dict]:
    return provenance.list_runs(limit)


def tail_log(run_id: str, lines: int = 100) -> list[str]:
    log = _run_dir(run_id) / "output.log"
    if not log.exists():
        return []
    text = log.read_text()
    return text.splitlines()[-lines:]


def _output_dirs(run_id: str) -> list[Path]:
    """Directories that may hold a run's output: demo runs write under runs/<id>/output,
    LIVE runs write to a model-specific shared dir recorded in job['output_dir']."""
    dirs = [_run_dir(run_id) / "output"]
    job = _jobs.get(run_id) or provenance.get_run(run_id) or {}
    od = job.get("output_dir")
    if od:
        dirs.append(Path(od))
    return dirs


def get_output_files(run_id: str) -> list[dict]:
    files = []
    for out_dir in _output_dirs(run_id):
        if not out_dir.exists():
            continue
        for f in sorted(out_dir.iterdir()):
            if f.is_file():
                files.append({"name": f.name, "size": f.stat().st_size, "path": str(f)})
    return files


def get_output_file_path(run_id: str, name: str) -> Path | None:
    """Resolve a single named output file, guarding against path traversal."""
    safe = os.path.basename(name)
    for out_dir in _output_dirs(run_id):
        p = out_dir / safe
        if p.is_file():
            return p
    return None


async def stream_log(run_id: str) -> AsyncIterator[str]:
    """Yield SSE events for a run log."""
    # First replay existing log
    for line in tail_log(run_id, 200):
        yield f"data: {json.dumps({'line': line})}\n\n"

    q = _queues.get(run_id)
    if q is None:
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    while True:
        try:
            item = await asyncio.wait_for(q.get(), timeout=30)
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'ping': True})}\n\n"
            continue
        if item is None:
            yield f"data: {json.dumps({'done': True})}\n\n"
            return
        yield f"data: {json.dumps({'line': item})}\n\n"
