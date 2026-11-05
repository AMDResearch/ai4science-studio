"""CLI for AI4Science Studio campaign runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai4s.campaign import CampaignError, load_campaign, stage_plan
from ai4s.graph import run_campaign
from ai4s.paths import cluster_registry_dir, repo_root_from
from ai4s.registry import merge_agent_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai4s", description="AI4Science Studio campaign runtime")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a campaign.yaml through LangGraph")
    p_run.add_argument("campaign", type=Path, help="Path to campaign.yaml")
    p_run.add_argument("--run-dir", type=Path, default=None)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument(
        "-s",
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a campaign input (repeatable)",
    )

    p_val = sub.add_parser("validate", help="Validate campaigns and reusable agents")
    p_val.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional campaign.yaml paths (default: all under campaigns/)",
    )

    p_agents = sub.add_parser("agents", help="List reusable agents (repo + cluster registry)")

    args = parser.parse_args(argv)
    root = repo_root_from()

    if args.cmd == "run":
        overrides = _parse_sets(args.set)
        try:
            out = run_campaign(
                args.campaign,
                overrides=overrides,
                run_dir=args.run_dir,
                root=root,
                dry_run=args.dry_run,
            )
        except CampaignError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({k: v for k, v in out.items() if k != "stage_results"}, indent=2))
        status = out.get("status")
        if status in {"succeeded", "partial"} or out.get("dry_run"):
            return 0
        return 1

    if args.cmd == "validate":
        return _cmd_validate(root, args.paths)

    if args.cmd == "agents":
        agents = merge_agent_registry(root)
        reg = cluster_registry_dir(root)
        print(f"cluster_registry: {reg if reg is not None else '(unset)'}")
        for name, agent in sorted(agents.items()):
            print(f"{name}\t{agent.get('version')}\t{agent.get('_path')}")
        return 0

    return 2


def _parse_sets(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --set {item!r}; expected KEY=VALUE")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _cmd_validate(root: Path, paths: list[Path]) -> int:
    errors: list[str] = []
    # Agents
    try:
        agents = merge_agent_registry(root)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"agent registry: {exc}")
        agents = {}
    if not agents:
        # Not fatal for MVP if catalog is empty during bootstrap, but warn.
        print("warning: no reusable agents found under agents/", file=sys.stderr)

    # Campaigns
    if paths:
        campaign_files = paths
    else:
        campaign_files = sorted((root / "campaigns").glob("*/campaign.yaml"))
    for path in campaign_files:
        try:
            camp = load_campaign(path, root)
            stage_plan(camp)
        except CampaignError as exc:
            errors.append(f"{path}: {exc}")

    # Schema self-check on fixtures if present
    for schema_name in (
        "campaign.schema.json",
        "run.schema.json",
        "result.schema.json",
        "findings.schema.json",
        "agent.schema.json",
    ):
        schema_path = root / "schemas" / schema_name
        if not schema_path.is_file():
            errors.append(f"missing schema: {schema_name}")

    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"ok: {len(campaign_files)} campaign(s), {len(agents)} agent(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
