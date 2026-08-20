#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "detect_lab_provider_stats_mismatches.py"


def load_script():
    spec = importlib.util.spec_from_file_location("detect_lab_provider_stats_mismatches", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


detector = load_script()


class DetectLabProviderStatsMismatchesTest(unittest.TestCase):
    def test_finds_model_reported_by_stats_when_lab_provider_lacks_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            self._write(root / "models/alibaba/qwen3.8-2.4t-a95b.toml", "name = 'Qwen'\n")
            self._write(root / "providers/alibaba/provider.toml", "name = 'Alibaba'\n")
            for source, reported_provider in (
                ("openrouter", "alibaba"),
                ("vercel", "togetherai"),
                ("llmgateway", "deepinfra"),
            ):
                self._write(
                    root / f"stats/{source}/{reported_provider}/models/alibaba/qwen3.8-2.4t-a95b.json",
                    json.dumps({"uptime_last_1d": 99.5}) + "\n",
                )

            result = detector.scan_repository(root)

        self.assertEqual(result.errors, ())
        self.assertEqual(result.reported_model_count, 1)
        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.model_id, "alibaba/qwen3.8-2.4t-a95b")
        self.assertEqual(finding.lab_provider, "alibaba")
        self.assertTrue(finding.canonical_model_exists)
        self.assertTrue(finding.lab_provider_exists)
        self.assertFalse(finding.lab_provider_model_exists)
        self.assertEqual(
            {(report.source, report.reported_provider) for report in finding.reports},
            {("openrouter", "alibaba"), ("vercel", "togetherai"), ("llmgateway", "deepinfra")},
        )

    def test_accepts_provider_model_with_different_path_when_base_model_matches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            self._write(root / "models/alibaba/qwen3.8-2.4t-a95b.toml", "name = 'Qwen'\n")
            self._write(root / "providers/alibaba/provider.toml", "name = 'Alibaba'\n")
            self._write(
                root / "providers/alibaba/models/qwen3.8-open.toml",
                'base_model = "alibaba/qwen3.8-2.4t-a95b"\n',
            )
            self._write(
                root / "stats/openrouter/alibaba/models/alibaba/qwen3.8-2.4t-a95b.json",
                json.dumps({"uptime_last_1d": 99.5}) + "\n",
            )

            result = detector.scan_repository(root)

        self.assertEqual(result.errors, ())
        self.assertEqual(result.findings, ())

    def test_reports_missing_lab_provider_and_malformed_stats_separately(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            self._write(root / "stats/vercel/openai/models/acme/model.json", "not-json\n")
            self._write(
                root / "stats/openrouter/openai/models/acme/model.json",
                json.dumps({"uptime_last_1d": 99}) + "\n",
            )

            result = detector.scan_repository(root)

        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.model_id, "acme/model")
        self.assertEqual(finding.lab_provider, "acme")
        self.assertFalse(finding.canonical_model_exists)
        self.assertFalse(finding.lab_provider_exists)
        self.assertFalse(finding.lab_provider_model_exists)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("stats/vercel/openai/models/acme/model.json", result.errors[0])

    def test_ignores_json_files_without_stats_model_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            self._write(root / "stats/openrouter/provider.json", json.dumps({"data": []}))
            self._write(root / "stats/openrouter/provider/not-models/model.json", json.dumps({}))

            result = detector.scan_repository(root)

        self.assertEqual(result.findings, ())
        self.assertEqual(result.errors, ())
        self.assertEqual(result.reported_model_count, 0)

    @staticmethod
    def _write(path: pathlib.Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
