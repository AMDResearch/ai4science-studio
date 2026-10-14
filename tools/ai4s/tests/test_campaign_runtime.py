"""No-GPU unit tests for the model-agnostic campaign runtime."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai4s.campaign import CampaignError, load_campaign, resolve_inputs, stage_plan
from ai4s.graph import build_graph, run_campaign
from ai4s.llm import resolve_llm_config
from ai4s.paths import repo_root_from
from ai4s.records import new_result, validate_required_artifacts
from ai4s.registry import get_agent, get_task, merge_agent_registry
from ai4s.runner import RunnerError, build_argv
from ai4s.stages import can_continue


class SchemaAndCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = repo_root_from()
        self.campaign = self.root / "campaigns" / "orbit2-infer-perf" / "campaign.yaml"

    def test_campaign_loads(self) -> None:
        data = load_campaign(self.campaign, self.root)
        stages = stage_plan(data)
        self.assertEqual([s["id"] for s in stages], ["orbit_inference", "perf_report"])
        self.assertEqual(stages[0]["ref"], "ORBIT-2:inference-perf")
        self.assertEqual(stages[1]["ref"], "perf-report")

    def test_resolve_inputs_requires_shared(self) -> None:
        data = load_campaign(self.campaign, self.root)
        cleared = {
            "AI4S_SHARED_DIR": "",
            "PERF_TOOLS_DIR": "",
            "ORBIT2_ROOT": "",
        }
        with mock.patch.dict(os.environ, cleared, clear=False):
            for k in cleared:
                os.environ.pop(k, None)
            with self.assertRaises(CampaignError):
                resolve_inputs(data, {})

    def test_resolve_inputs_ok(self) -> None:
        data = load_campaign(self.campaign, self.root)
        resolved = resolve_inputs(
            data,
            {
                "AI4S_SHARED_DIR": "/tmp/shared",
                "PERF_TOOLS_DIR": "/tmp/perf",
                "ORBIT2_ROOT": "/tmp/orbit2",
            },
        )
        self.assertEqual(resolved["AI4S_SHARED_DIR"], "/tmp/shared")
        self.assertEqual(resolved["ORBIT2_USE_SYNTHETIC"], "1")


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = repo_root_from()

    def test_perf_report_agent(self) -> None:
        agent = get_agent("perf-report", self.root)
        self.assertEqual(agent["kind"], "agent")
        self.assertFalse((agent.get("allow") or {}).get("submit_jobs", False))

    def test_orbit_task_is_model_local(self) -> None:
        task = get_task("ORBIT-2:inference-perf", self.root)
        self.assertIn("ORBIT-2", task["model_dir"])
        self.assertTrue(str(task["script"]).endswith("run_infer_perf_amd.sh"))

    def test_merge_registry_contains_perf_report(self) -> None:
        agents = merge_agent_registry(self.root)
        self.assertIn("perf-report", agents)


class RunnerAndGatingTests(unittest.TestCase):
    def test_refuses_sbatch(self) -> None:
        with self.assertRaises(RunnerError):
            build_argv(["sbatch", "foo.sh"], use_srun=False)

    def test_srun_prefix_in_allocation(self) -> None:
        argv = build_argv(["echo", "hi"], use_srun=True)
        self.assertEqual(argv[:2], ["srun", "--export=ALL"])

    def test_failure_blocks_next(self) -> None:
        stage = {"id": "a", "required_for_next": ["manifest"]}
        bad = new_result(run_id="r", name="a", status="error")
        self.assertFalse(can_continue(bad, stage))

    def test_partial_requires_artifacts(self) -> None:
        stage = {"id": "a", "required_for_next": ["manifest"]}
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            ok = new_result(
                run_id="r",
                name="a",
                status="partial",
                artifacts={"manifest": {"path": str(manifest), "sha256": None}},
            )
            self.assertTrue(can_continue(ok, stage))
            missing = new_result(run_id="r", name="a", status="partial", artifacts={})
            self.assertFalse(can_continue(missing, stage))

    def test_validate_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "foms.json"
            p.write_text("{}", encoding="utf-8")
            result = {
                "artifacts": {
                    "foms": {"path": str(p)},
                    "missing": {"path": str(Path(tmp) / "nope")},
                }
            }
            ok, problems = validate_required_artifacts(result, ["foms", "missing"])
            self.assertFalse(ok)
            self.assertTrue(any("missing" in x for x in problems))


class LLMConfigTests(unittest.TestCase):
    def test_modes(self) -> None:
        cfg = resolve_llm_config({"mode": "external", "endpoint": "http://x/v1"})
        self.assertEqual(cfg["mode"], "external")
        self.assertTrue(cfg["endpoint"])

    def test_amd_apim_auth_headers(self) -> None:
        from ai4s.llm import LLMEndpoint, auth_headers, _resolved_auth_style

        ep = LLMEndpoint(
            mode="external",
            base_url="https://llm-api.amd.com/OpenAI/v1",
            model="Claude-Sonnet-4.6",
            api_key_env="OPENAI_API_KEY",
            auth_style="auto",
        )
        self.assertEqual(_resolved_auth_style(ep), "apim")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-sub-key"}, clear=False):
            headers = auth_headers(ep)
        self.assertEqual(headers.get("Ocp-Apim-Subscription-Key"), "test-sub-key")
        self.assertNotIn("Authorization", headers)

    def test_bearer_auth_headers(self) -> None:
        from ai4s.llm import LLMEndpoint, auth_headers

        ep = LLMEndpoint(
            mode="external",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            auth_style="auto",
        )
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
            headers = auth_headers(ep)
        self.assertEqual(headers.get("Authorization"), "Bearer sk-test")


class GraphDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = repo_root_from()
        self.campaign = self.root / "campaigns" / "orbit2-infer-perf" / "campaign.yaml"

    def test_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = run_campaign(
                self.campaign,
                overrides={
                    "AI4S_SHARED_DIR": tmp,
                    "PERF_TOOLS_DIR": tmp,
                    "ORBIT2_ROOT": tmp,
                },
                run_dir=Path(tmp) / "run",
                root=self.root,
                dry_run=True,
            )
            self.assertTrue(out["dry_run"])
            self.assertEqual(out["stages"], ["orbit_inference", "perf_report"])

    def test_graph_builds(self) -> None:
        app = build_graph()
        self.assertIsNotNone(app)

    def test_failed_first_stage_skips_second(self) -> None:
        """Mock execute_stage so first fails; second must not run."""
        from ai4s import graph as graph_mod

        calls: list[str] = []

        def fake_execute(stage, **kwargs):
            calls.append(stage["id"])
            return new_result(
                run_id=kwargs["run_id"],
                name=stage["id"],
                status="error",
                error={"type": "Boom", "message": "fail"},
            )

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(graph_mod, "execute_stage", side_effect=fake_execute):
                out = run_campaign(
                    self.campaign,
                    overrides={
                        "AI4S_SHARED_DIR": tmp,
                        "PERF_TOOLS_DIR": tmp,
                        "ORBIT2_ROOT": tmp,
                    },
                    run_dir=Path(tmp) / "run",
                    root=self.root,
                    dry_run=False,
                )
        self.assertEqual(out["status"], "failed")
        self.assertEqual(calls, ["orbit_inference"])


class PerfReportAgentTests(unittest.TestCase):
    def test_deterministic_on_fixture(self) -> None:
        from ai4s.agents.perf_report import run as perf_run

        root = repo_root_from()
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job"
            job.mkdir()
            (job / "manifest.json").write_text(
                json.dumps({"state": "ok", "inference_wall_seconds": 12.5, "job_id": "1"}),
                encoding="utf-8",
            )
            (job / "foms.json").write_text(
                json.dumps({"inference_wall_seconds": 12.5}),
                encoding="utf-8",
            )
            (job / "omnistat-db").mkdir()
            stage_dir = Path(tmp) / "stage"
            result = perf_run(
                {
                    "run_id": "t1",
                    "stage_id": "perf_report",
                    "stage_dir": stage_dir,
                    "resolved_inputs": {"_campaign_llm": {"mode": "external", "endpoint": ""}},
                    "prior_artifacts": {
                        "manifest": {"path": str(job / "manifest.json")},
                    },
                    "agent": {"name": "perf-report"},
                    "root": root,
                    "env": {},
                }
            )
            # LLM provision fails without endpoint → partial with deterministic findings
            self.assertIn(result["status"], {"pass", "partial"})
            self.assertTrue((stage_dir / "findings.json").is_file())
            self.assertTrue((stage_dir / "perf_report.md").is_file())
            findings = json.loads((stage_dir / "findings.json").read_text(encoding="utf-8"))
            self.assertTrue(findings["findings"])


if __name__ == "__main__":
    unittest.main()
