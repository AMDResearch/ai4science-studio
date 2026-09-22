from __future__ import annotations

import unittest
from pathlib import Path

from ai4s_validate.discover import ModelEntry
from ai4s_validate.images import _iter_images, _ok_image, check_container_images


class TestImageRefs(unittest.TestCase):
    def test_known_good(self) -> None:
        self.assertTrue(_ok_image("rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0"))
        self.assertTrue(_ok_image("jaxweather:latest"))
        self.assertTrue(_ok_image("pytorchweather:latest"))
        self.assertTrue(_ok_image("rocm/dev-ubuntu-22.04:7.0.2-complete"))
        self.assertTrue(_ok_image("images/model.sif"))

    def test_bad(self) -> None:
        self.assertFalse(_ok_image("rocm/pytorch"))
        self.assertFalse(_ok_image("not a tag"))
        self.assertFalse(_ok_image(""))

    def test_iter_list(self) -> None:
        self.assertEqual(_iter_images([" a:b ", ""]), ["a:b"])

    def test_malformed_is_error(self) -> None:
        model = ModelEntry(
            slug="Fake",
            domain="earth_science",
            path=Path("earth_science/models/Fake"),
            manifest_path=Path("earth_science/models/Fake/model.yaml"),
            data={"container_image": "no-tag"},
        )
        findings = check_container_images(Path("."), [model]).findings
        self.assertTrue(any(f.code == "container-image-format" for f in findings), findings)
