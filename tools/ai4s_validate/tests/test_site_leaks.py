from __future__ import annotations

import unittest

from ai4s_validate.site_leaks import scan_text


class TestSiteLeaks(unittest.TestCase):
    def test_shared_username_is_error(self) -> None:
        hits = scan_text('HG_SIF="${HG_SIF:-/shared/aaji/images/foo.sif}"', rel="x.sh", model="HydraGNN")
        self.assertTrue(any(f.code == "site-path" for f in hits), hits)

    def test_shared_user_placeholder_ok(self) -> None:
        hits = scan_text("export AI4S_SHARED_DIR=/shared/$USER", rel="x.sh", model="HydraGNN")
        self.assertEqual(hits, [])

    def test_your_shared_dir_ok(self) -> None:
        hits = scan_text(
            'SIF="${AI4S_SHARED_DIR:-/your/shared/dir}/images/x.sif"',
            rel="x.sh",
            model="StormCast",
        )
        self.assertEqual(hits, [])

    def test_placeholder_partition_ok(self) -> None:
        text = "#SBATCH --partition=YOUR_PARTITION_HERE\n#SBATCH --account=YOUR_ACCOUNT_HERE\n"
        self.assertEqual(scan_text(text, rel="s.sh", model="x"), [])

    def test_real_partition_is_error(self) -> None:
        hits = scan_text("#SBATCH --partition=mi300x\n", rel="s.sh", model="x")
        self.assertTrue(any(f.code == "site-partition" for f in hits), hits)

    def test_commented_frontier_path_ok(self) -> None:
        hits = scan_text(
            '#    \'ERA5_1\': "/lustre/orion/lrn036/world-shared/data/"\n',
            rel="cfg.yaml",
            model="ORBIT-2",
        )
        self.assertEqual(hits, [])

    def test_docstring_job_example_ok(self) -> None:
        src = '''"""Usage:
    python parse.py --log hydragnn-train-6294.out
"""
print("ok")
'''
        hits = scan_text(src, rel="parse_convergence.py", model="HydraGNN")
        self.assertEqual(hits, [])

    def test_active_job_id_is_error(self) -> None:
        hits = scan_text('LOG = "hydragnn-train-6294.out"\n', rel="x.py", model="HydraGNN")
        self.assertTrue(any(f.code == "site-job-id" for f in hits), hits)

    def test_home_path_is_error(self) -> None:
        hits = scan_text("cd /home/aaji/git/ai4science-studio\n", rel="x.sh", model="x")
        self.assertTrue(any(f.code == "site-path" for f in hits), hits)


if __name__ == "__main__":
    unittest.main()
