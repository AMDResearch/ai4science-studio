"""Build and run a LangGraph StateGraph from a campaign file.

The graph structure is derived from ``campaign['stages']`` — this module does
not hardcode campaign- or model-specific node names.
"""

from __future__ import annotations

import traceback
import uuid
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ai4s.campaign import load_campaign, resolve_inputs, stage_plan
from ai4s.llm import LLMEndpoint, resolve_llm_config, stop_local_llm
from ai4s.paths import repo_root_from
from ai4s.records import finalize_run, new_run_record, write_json
from ai4s.stages import can_continue, execute_stage


class CampaignState(TypedDict, total=False):
    run_id: str
    campaign_name: str
    run_dir: str
    root: str
    stages: list[dict[str, Any]]
    stage_index: int
    resolved_inputs: dict[str, Any]
    stage_results: dict[str, dict[str, Any]]
    artifacts: dict[str, Any]
    status: str  # running | succeeded | failed | partial
    error: dict[str, Any] | None
    llm_meta: dict[str, Any]
    run_record: dict[str, Any]


def build_graph() -> Any:
    """Compile a generic sequential campaign graph with failure gating."""
    graph = StateGraph(CampaignState)
    graph.add_node("run_stage", _node_run_stage)
    graph.add_node("advance", _node_advance)
    graph.add_node("succeed", _node_succeed)
    graph.add_node("fail", _node_fail)

    graph.add_edge(START, "run_stage")
    graph.add_conditional_edges(
        "run_stage",
        _route_after_stage,
        {
            "advance": "advance",
            "succeed": "succeed",
            "fail": "fail",
        },
    )
    graph.add_conditional_edges(
        "advance",
        _route_after_advance,
        {
            "run_stage": "run_stage",
            "succeed": "succeed",
        },
    )
    graph.add_edge("succeed", END)
    graph.add_edge("fail", END)
    return graph.compile(checkpointer=MemorySaver())


def _current_stage(state: CampaignState) -> dict[str, Any]:
    return state["stages"][state["stage_index"]]


def _node_run_stage(state: CampaignState) -> dict[str, Any]:
    stage = _current_stage(state)
    run_dir = Path(state["run_dir"])
    root = Path(state["root"])
    try:
        result = execute_stage(
            stage,
            run_id=state["run_id"],
            run_dir=run_dir,
            resolved_inputs=state.get("resolved_inputs") or {},
            root=root,
            prior_artifacts=state.get("artifacts") or {},
        )
    except Exception as exc:  # noqa: BLE001 — convert to terminal stage error
        result = {
            "schema_version": "1",
            "run_id": state["run_id"],
            "name": stage["id"],
            "status": "error",
            "checks": [],
            "metrics": {},
            "artifacts": {},
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            },
        }

    stage_results = dict(state.get("stage_results") or {})
    stage_results[stage["id"]] = result
    artifacts = dict(state.get("artifacts") or {})
    for key, entry in (result.get("artifacts") or {}).items():
        artifacts[f"{stage['id']}.{key}"] = entry
        artifacts[key] = entry

    run_record = dict(state.get("run_record") or {})
    nodes = dict(run_record.get("nodes") or {})
    nodes[stage["id"]] = {
        "status": result.get("status"),
        "result_path": str(run_dir / "stages" / stage["id"] / "result.json"),
    }
    run_record["nodes"] = nodes

    updates: dict[str, Any] = {
        "stage_results": stage_results,
        "artifacts": artifacts,
        "run_record": run_record,
    }
    if not can_continue(result, stage):
        updates["status"] = "failed"
        updates["error"] = {
            "type": "StageFailed",
            "message": (result.get("error") or {}).get("message")
            or f"stage {stage['id']} status={result.get('status')}",
            "stage_id": stage["id"],
        }
    return updates


def _route_after_stage(state: CampaignState) -> str:
    if state.get("status") == "failed":
        return "fail"
    stage = _current_stage(state)
    result = (state.get("stage_results") or {})[stage["id"]]
    if not can_continue(result, stage):
        return "fail"
    if state["stage_index"] >= len(state["stages"]) - 1:
        return "succeed"
    return "advance"


def _node_advance(state: CampaignState) -> dict[str, Any]:
    return {"stage_index": int(state["stage_index"]) + 1}


def _route_after_advance(state: CampaignState) -> str:
    if state["stage_index"] >= len(state["stages"]):
        return "succeed"
    return "run_stage"


def _node_succeed(state: CampaignState) -> dict[str, Any]:
    # Campaign is partial if any stage was partial.
    statuses = [r.get("status") for r in (state.get("stage_results") or {}).values()]
    final = "partial" if "partial" in statuses else "succeeded"
    run_record = finalize_run(dict(state.get("run_record") or {}), state=final, error=None)
    write_json(Path(state["run_dir"]) / "run.json", run_record)
    return {"status": final, "run_record": run_record, "error": None}


def _node_fail(state: CampaignState) -> dict[str, Any]:
    error = state.get("error") or {
        "type": "CampaignFailed",
        "message": "campaign failed",
    }
    run_record = finalize_run(
        dict(state.get("run_record") or {}),
        state="failed",
        error=error,
    )
    write_json(Path(state["run_dir"]) / "run.json", run_record)
    return {"status": "failed", "run_record": run_record, "error": error}


def run_campaign(
    campaign_path: Path,
    *,
    overrides: dict[str, Any] | None = None,
    run_dir: Path | None = None,
    root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Load campaign YAML, build graph state, execute with failure gating."""
    root = root or repo_root_from()
    campaign = load_campaign(campaign_path, root)
    stages = stage_plan(campaign)
    resolved = resolve_inputs(campaign, overrides)
    run_id = resolved.get("run_id") or f"{campaign['name']}-{uuid.uuid4().hex[:10]}"
    if run_dir is None:
        shared = resolved.get("AI4S_SHARED_DIR") or resolved.get("shared_dir")
        if shared:
            run_dir = Path(str(shared)) / "campaign-runs" / str(run_id)
        else:
            run_dir = root / "runs" / "campaigns" / str(run_id)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = resolve_llm_config(campaign.get("llm"))
    # Do not put secrets into the run record — only env var *names* and mode.
    llm_meta = {
        "mode": llm_cfg["mode"],
        "endpoint_configured": bool(llm_cfg.get("endpoint")),
        "api_key_env": llm_cfg.get("api_key_env"),
        "model": llm_cfg.get("model"),
    }
    run_record = new_run_record(
        run_id=str(run_id),
        name=campaign["name"],
        kind="campaign",
        resolved_inputs=_public_inputs(resolved),
        repo=root,
        llm=llm_meta,
    )
    write_json(run_dir / "run.json", run_record)
    write_json(run_dir / "campaign.resolved.json", {"stages": stages, "inputs": _public_inputs(resolved)})

    if dry_run:
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": "queued",
            "stages": [s["id"] for s in stages],
            "dry_run": True,
        }

    # Provision LLM only when an agent stage exists; after science tasks ideally.
    # For MVP we provision lazily inside the agent. Keep handle for cleanup.
    llm_endpoint: LLMEndpoint | None = None
    initial: CampaignState = {
        "run_id": str(run_id),
        "campaign_name": campaign["name"],
        "run_dir": str(run_dir),
        "root": str(root),
        "stages": stages,
        "stage_index": 0,
        "resolved_inputs": {**resolved, "_campaign_llm": campaign.get("llm") or {}},
        "stage_results": {},
        "artifacts": {},
        "status": "running",
        "error": None,
        "llm_meta": llm_meta,
        "run_record": run_record,
    }

    app = build_graph()
    config = {"configurable": {"thread_id": str(run_id)}}
    try:
        final_state = app.invoke(initial, config=config)
    finally:
        stop_local_llm(llm_endpoint)

    # Write campaign-level result.json
    from ai4s.records import new_result

    status_map = {"succeeded": "pass", "partial": "partial", "failed": "error"}
    camp_status = status_map.get(final_state.get("status", "failed"), "error")
    result = new_result(
        run_id=str(run_id),
        name=campaign["name"],
        status=camp_status,
        artifacts=final_state.get("artifacts") or {},
        error=final_state.get("error"),
        checks=[
            {
                "name": f"stage:{sid}",
                "status": "pass"
                if (final_state.get("stage_results") or {}).get(sid, {}).get("status")
                in {"pass", "partial"}
                else "fail",
                "detail": (final_state.get("stage_results") or {})
                .get(sid, {})
                .get("status", "missing"),
            }
            for sid in [s["id"] for s in stages]
        ],
    )
    write_json(run_dir / "result.json", result)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": final_state.get("status"),
        "result": result,
        "stage_results": final_state.get("stage_results") or {},
    }


def _public_inputs(resolved: dict[str, Any]) -> dict[str, Any]:
    """Drop values that look like secrets from the persisted input snapshot."""
    out: dict[str, Any] = {}
    for k, v in resolved.items():
        lk = k.lower()
        if any(s in lk for s in ("api_key", "token", "password", "secret")):
            out[k] = "***"
        else:
            out[k] = v
    return out
