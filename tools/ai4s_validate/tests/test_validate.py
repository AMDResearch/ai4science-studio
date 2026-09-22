from __future__ import annotations

import unittest
from pathlib import Path

from ai4s_validate.discover import discover_models, repo_root_from
from ai4s_validate.index import build_index_entries, expected_index_text
from ai4s_validate.manifests import check_manifests, load_schema
from ai4s_validate.runner import run_fast


class TestRepoRoot(unittest.TestCase):
    def test_root_has_schema(self) -> None:
        root = repo_root_from()
        self.assertTrue((root / "schemas" / "model.schema.json").is_file())
        self.assertTrue((root / "earth_science").is_dir())


class TestDiscover(unittest.TestCase):
    def test_finds_fifteen_models(self) -> None:
        root = repo_root_from()
        models = discover_models(root)
        slugs = {m.slug for m in models}
        self.assertGreaterEqual(len(models), 15, slugs)
        self.assertIn("ORBIT-2", slugs)
        self.assertIn("HydraGNN", slugs)
        self.assertIn("GP-MoLFormer", slugs)


class TestSchema(unittest.TestCase):
    def test_all_manifests_match_schema(self) -> None:
        root = repo_root_from()
        models = discover_models(root)
        result = check_manifests(root, models)
        schema_errors = [f for f in result.findings if f.code == "schema"]
        self.assertEqual(schema_errors, [], schema_errors)
        load_schema(root)  # must parse


class TestIndex(unittest.TestCase):
    def test_generated_index_is_stable(self) -> None:
        root = repo_root_from()
        models = discover_models(root)
        text = expected_index_text(models)
        self.assertIn("GENERATED FILE", text)
        entries = build_index_entries(models)
        slugs = [e["slug"] for e in entries]
        self.assertEqual(len(slugs), len(set(slugs)))
        self.assertIn("StormCast", slugs)

    def test_run_fast_after_generation(self) -> None:
        root = repo_root_from()
        from ai4s_validate.index import write_index

        write_index(root, discover_models(root))
        result = run_fast(root)
        self.assertEqual(result.errors, [], result.errors)


if __name__ == "__main__":
    unittest.main()
