"""Execute a single campaign stage (task or agent) with I/O validation."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from ai4s.records import (
    artifact_entry,
    new_result,
    validate_required_artifacts,
    write_json,
)
from ai4s.registry import RegistryError, get_agent, get_task
from ai4s.runner import RunnerError, run_command


StageFn = Callable[[dict[str, Any]], dict[str, Any]]


def execute_stage(
    stage: dict[str, Any],
    *,
    run_id: str,
    run_dir: Path,
    resolved_inputs: dict[str, Any],
    root: Path,
    prior_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one stage and return a result.json-compatible dict."""
    stage_id = stage["id"]
    stage_dir = run_dir / "stages" / stage_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    kind = stage["kind"]
    try:
        if kind == "task":
            result = _run_task(
                stage,
                run_id=run_id,
                stage_dir=stage_dir,
                resolved_inputs=resolved_inputs,
                root=root,
            )
        elif kind == "agent":
            result = _run_agent(
                stage,
                run_id=run_id,
                stage_dir=stage_dir,
                resolved_inputs=resolved_inputs,
                root=root,
                prior_artifacts=prior_artifacts or {},
            )
        else:
            result = new_result(
                run_id=run_id,
                name=stage_id,
                status="error",
                error={"type": "InvalidStage", "message": f"unknown kind: {kind}"},
            )
    except (RegistryError, RunnerError, OSError, ValueError) as exc:
        result = new_result(
            run_id=run_id,
            name=stage_id,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )

    required = list(stage.get("required_for_next") or [])
    if result["status"] in {"pass", "partial"} and required:
        ok, problems = validate_required_artifacts(result, required)
        if not ok:
            result = new_result(
                run_id=run_id,
                name=stage_id,
                status="error",
                checks=result.get("checks") or [],
                metrics=result.get("metrics") or {},
                artifacts=result.get("artifacts") or {},
                error={
                    "type": "MissingRequiredArtifacts",
                    "message": "; ".join(problems),
                },
            )

    write_json(stage_dir / "result.json", result)
    return result


def can_continue(result: dict[str, Any], stage: dict[str, Any]) -> bool:
    """Whether the next stage may run given this stage's result."""
    status = result.get("status")
    if status == "pass":
        return True
    if status == "partial":
        required = list(stage.get("required_for_next") or [])
        ok, _ = validate_required_artifacts(result, required)
        return ok
    return False


def _run_task(
    stage: dict[str, Any],
    *,
    run_id: str,
    stage_dir: Path,
    resolved_inputs: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    task = get_task(stage["ref"], root)
    command = stage.get("command")
    if command:
        script = (root / command).resolve() if not Path(command).is_absolute() else Path(command)
    else:
        # Prefer allocation-neutral perf/infer scripts when present.
        model_dir = Path(task["model_dir"])
        candidates = []
        if task.get("script"):
            candidates.append(model_dir / task["script"])
        # Fallbacks for older manifests that still point at run_visualize.py
        if task["task"] in {"inference", "inference-perf"}:
            candidates.append(model_dir / "examples" / "run_infer_perf_amd.sh")
            candidates.append(model_dir / "examples" / "run_infer_amd.sh")
        script = next((c for c in candidates if c.is_file()), None)
        if script is None:
            raise RegistryError(f"no runnable script for task {stage['ref']}")

    env = {
        **{k: str(v) for k, v in resolved_inputs.items()},
        **{k: str(v) for k, v in (stage.get("env") or {}).items()},
        "AI4S_RUN_ID": run_id,
        "AI4S_STAGE_DIR": str(stage_dir),
        "AI4S_OUTPUT_DIR": str(stage_dir / "artifacts"),
    }
    # Common alias used by ORBIT perf scripts.
    env.setdefault("ORBIT2_OUTPUT_DIR", env["AI4S_OUTPUT_DIR"])
    (stage_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # Task scripts are self-contained SLURM-aware shells; do not wrap in srun
    # (they already call srun internally for ranks).
    cmd_result = run_command(
        ["bash", str(script)],
        cwd=script.parent,
        env=env,
        use_srun=False,
        log_path=stage_dir / "command.log",
    )

    artifacts: dict[str, Any] = {}
    art_dir = Path(env["AI4S_OUTPUT_DIR"])
    for name in ("manifest.json", "foms.json", "omnistat-db", "omnistat.config"):
        path = art_dir / name
        if path.exists():
            artifacts[name.replace(".json", "").replace("-", "_")] = artifact_entry(
                path,
                required_for_next=name in {"manifest.json", "foms.json"},
            )
    # Also accept result.json written by the task itself.
    task_result_path = art_dir / "result.json"
    if task_result_path.is_file():
        task_result = json.loads(task_result_path.read_text(encoding="utf-8"))
        if isinstance(task_result, dict) and "status" in task_result:
            task_result.setdefault("run_id", run_id)
            task_result.setdefault("name", stage["id"])
            task_result.setdefault("schema_version", "1")
            return task_result

    status = "pass" if cmd_result.returncode == 0 else "error"
    checks = [
        {
            "name": "exit_status",
            "status": "pass" if cmd_result.returncode == 0 else "fail",
            "detail": f"returncode={cmd_result.returncode}",
        }
    ]
    if status == "pass" and "manifest" not in artifacts:
        # Soft partial when command succeeded but telemetry missing.
        status = "partial"
        checks.append(
            {
                "name": "manifest_present",
                "status": "fail",
                "detail": "manifest.json not produced",
            }
        )
    return new_result(
        run_id=run_id,
        name=stage["id"],
        status=status,
        checks=checks,
        artifacts=artifacts,
        error=None
        if status != "error"
        else {
            "type": "CommandFailed",
            "message": f"exit {cmd_result.returncode}; see {stage_dir / 'command.log'}",
        },
    )


def _run_agent(
    stage: dict[str, Any],
    *,
    run_id: str,
    stage_dir: Path,
    resolved_inputs: dict[str, Any],
    root: Path,
    prior_artifacts: dict[str, Any],
) -> dict[str, Any]:
    agent = get_agent(stage["ref"], root)
    allow = agent.get("allow") or {}
    if allow.get("submit_jobs") is True:
        raise RegistryError(f"agent {stage['ref']} must not submit jobs in this MVP")

    entry = agent.get("entry") or "ai4s.agents.perf_report:run"
    fn = _load_entry(entry, agent_dir=Path(agent["_dir"]))
    context = {
        "run_id": run_id,
        "stage_id": stage["id"],
        "stage_dir": stage_dir,
        "resolved_inputs": resolved_inputs,
        "prior_artifacts": prior_artifacts,
        "agent": agent,
        "root": root,
        "env": {k: str(v) for k, v in (stage.get("env") or {}).items()},
    }
    result = fn(context)
    if not isinstance(result, dict) or "status" not in result:
        raise RegistryError(f"agent {stage['ref']} returned invalid result")
    result.setdefault("schema_version", "1")
    result.setdefault("run_id", run_id)
    result.setdefault("name", stage["id"])
    return result


def _load_entry(entry: str, *, agent_dir: Path) -> StageFn:
    if ":" in entry:
        mod_name, fn_name = entry.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        if not callable(fn):
            raise RegistryError(f"entry not callable: {entry}")
        return fn  # type: ignore[return-value]
    script = Path(entry)
    if not script.is_absolute():
        script = agent_dir / script
    if not script.is_file():
        raise RegistryError(f"agent entry script not found: {script}")

    def _run(context: dict[str, Any]) -> dict[str, Any]:
        payload_path = Path(context["stage_dir"]) / "agent_context.json"
        write_json(payload_path, {k: str(v) if isinstance(v, Path) else v for k, v in context.items() if k != "agent"})
        env = os.environ.copy()
        env["AI4S_AGENT_CONTEXT"] = str(payload_path)
        cmd = run_command(
            ["python3", str(script)],
            cwd=agent_dir,
            env=env,
            use_srun=False,
            log_path=Path(context["stage_dir"]) / "agent.log",
        )
        out_path = Path(context["stage_dir"]) / "result.json"
        if out_path.is_file():
            return json.loads(out_path.read_text(encoding="utf-8"))
        if cmd.returncode != 0:
            return new_result(
                run_id=str(context["run_id"]),
                name=str(context["stage_id"]),
                status="error",
                error={"type": "AgentScriptFailed", "message": f"exit {cmd.returncode}"},
            )
        raise RegistryError(f"agent script produced no result.json: {script}")

    return _run
