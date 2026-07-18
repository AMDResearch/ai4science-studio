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


def _emit(run_id: str, line: str):
    """Append to log file and push to SSE queue if any listener."""
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
_SLURM_ACCOUNT   = "vultr_lux"
_SIF_PATH        = f"{_AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif"

# Per-model overlay paths (None = no overlay needed)
_OVERLAYS: dict[str, str | None] = {
    "HydraGNN":    f"{_AI4S_SHARED_DIR}/models/HydraGNN/overlays/hydragnn-overlay.img",
    "ORBIT-2":     f"{_AI4S_SHARED_DIR}/models/ORBIT-2/overlays/orbit2-overlay.img",
    "StormCast":   f"{_AI4S_SHARED_DIR}/models/StormCast/overlays/stormcast-overlay.img",
    "GP-MoLFormer": None,
}


def _build_slurm_script(run_id: str, slug: str, domain: str, task: str,
                         prompt: str, params: dict, partition: str) -> Path:
    """Build a minimal sbatch wrapper that echoes params and calls the model's sbatch script."""
    from registry import REPO_ROOT
    model_dir = REPO_ROOT / domain / "models" / slug
    log_path = str(_run_dir(run_id) / "slurm.log")

    # Pick the most specific sbatch script (prefer task-matching name).
    # HydraGNN uses our own trained-model inference script (studio_infer), which
    # predicts energy on a real held-out material and reports pred vs DFT.
    if slug == "HydraGNN":
        sbatch_candidates = list(model_dir.glob("examples/sbatch_studio_infer_amd.sh"))
    else:
        sbatch_candidates = (
            list(model_dir.glob(f"examples/sbatch*{task}*amd*.sh")) +
            list(model_dir.glob("examples/sbatch*infer*amd*.sh")) +
            list(model_dir.glob("examples/sbatch*amd*.sh"))
        )
    upstream_script = sbatch_candidates[0] if sbatch_candidates else None

    overlay = _OVERLAYS.get(slug)

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=studio-{slug[:10]}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --account={_SLURM_ACCOUNT}",
        "#SBATCH --nodes=1",
        "#SBATCH --gres=gpu:1",
        # One task per node so models that fan out via an inner `srun --mpi=pmix`
        # (e.g. ORBIT-2) get a bindable CPU mask. Do NOT pin --cpus-per-task: a
        # constrained mask makes the nested srun fail "Unable to satisfy cpu bind".
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --time=00:30:00",
        f"#SBATCH --output={log_path}",
        f"#SBATCH --error={log_path}",
        "",
        f"echo '[studio] Run ID: {run_id}'",
        f"echo '[studio] Model: {slug} | Task: {task}'",
        f"echo '[studio] Partition: {partition}'",
        f"echo '[studio] Prompt: {prompt[:100]}'",
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
    if slug == "ORBIT-2":
        script_lines += [
            f"export ORBIT2_ROOT=/shared/aaji/models/ORBIT-2/code/ORBIT-2",
            f"export ORBIT2_USE_SYNTHETIC=1",
            f"export ORBIT2_CONFIG=interm_8m_synthetic_1gpu.yaml",
            f"export ORBIT2_OUTPUT_DIR={_AI4S_SHARED_DIR}/models/ORBIT-2/outputs/{run_id}",
            "",
        ]

    # HydraGNN: predict energy with OUR trained PNA model on a real held-out material.
    _HG_WORK = f"{_AI4S_SHARED_DIR}/models/HydraGNN/train_work"
    if slug == "HydraGNN":
        # Pick checkpoint by variant: 8gpu (hg_model_ddp.pk) or 1gpu (hg_model.pk).
        variant = str(params.get("model_variant", "8gpu")).lower()
        ckpt = "hg_model_ddp.pk" if variant == "8gpu" else "hg_model.pk"
        struct_index = params.get("struct_index", params.get("STRUCT_INDEX", 4))
        script_lines += [
            f"export HG_INFER_REPO={_AI4S_SHARED_DIR}/models/HydraGNN/outputs/HydraGNN-infer",
            f"export HG_MODEL_PATH={_HG_WORK}/results/{ckpt}",
            f"export HG_MODEL_VARIANT={variant!r}",
            f"export STRUCT_INDEX={int(struct_index)}",
            f"export HG_STUDIO_INFER={_HG_WORK}/inference/studio_infer.py",
            f"export STUDIO_RESULT_OUT={str(_run_dir(run_id) / 'hg_result.json')}",
            "",
        ]

    # Extra params from UI
    for k, v in params.items():
        script_lines.append(f"export {k.upper()}={v!r}")
    script_lines.append("")

    if upstream_script:
        # Upstream scripts resolve their own dir via `scontrol show job`, which returns
        # THIS generated job.sh (not the examples dir) when we bash-call them. Pass the
        # real examples dir explicitly so their /examples bind-mount points at the scripts.
        script_lines.append(f"export STUDIO_EXAMPLES_DIR={str(upstream_script.parent)!r}")
        script_lines.append(f"echo '[studio] Delegating to upstream script: {upstream_script.name}'")
        script_lines.append(f"bash {upstream_script} 2>&1")
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
