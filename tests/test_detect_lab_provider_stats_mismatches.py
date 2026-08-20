#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import tomllib

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
            enabled_result = detector.scan_repository(root, detect_labs_provider_directory_missing=True)

        self.assertEqual(result.findings, ())
        self.assertEqual(len(enabled_result.findings), 1)
        finding = enabled_result.findings[0]
        self.assertEqual(finding.model_id, "acme/model")
        self.assertEqual(finding.lab_provider, "acme")
        self.assertFalse(finding.canonical_model_exists)
        self.assertFalse(finding.lab_provider_exists)
        self.assertFalse(finding.lab_provider_model_exists)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("stats/vercel/openai/models/acme/model.json", result.errors[0])

    def test_ignores_broken_provider_model_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            self._write(root / "models/alibaba/qwen.toml", "name = 'Qwen'\n")
            self._write(root / "providers/alibaba/provider.toml", "name = 'Alibaba'\n")
            models_dir = root / "providers/alibaba/models"
            models_dir.mkdir(parents=True)
            (models_dir / "broken.toml").symlink_to("missing.toml")
            self._write(
                root / "stats/openrouter/alibaba/models/alibaba/qwen.json",
                json.dumps({"uptime_last_1d": 99}) + "\n",
            )

            result = detector.scan_repository(root)

        self.assertEqual(result.errors, ())
        self.assertEqual(len(result.findings), 1)

    def test_directory_missing_detection_flag_defaults_to_false(self):
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH)]):
            self.assertFalse(detector.parse_args().detect_labs_provider_directory_missing)
        with mock.patch.object(
            sys,
            "argv",
            [str(SCRIPT_PATH), "--detect-labs-provider-directory-missing"],
        ):
            self.assertTrue(detector.parse_args().detect_labs_provider_directory_missing)

    def test_ignores_json_files_without_stats_model_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            self._write(root / "stats/openrouter/provider.json", json.dumps({"data": []}))
            self._write(root / "stats/openrouter/provider/not-models/model.json", json.dumps({}))

            result = detector.scan_repository(root)

        self.assertEqual(result.findings, ())
        self.assertEqual(result.errors, ())
        self.assertEqual(result.reported_model_count, 0)

    def test_writes_synthetic_provider_tree_for_missing_lab_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            self._write(root / "models/alibaba/qwen3.8-2.4t-a95b.toml", "name = 'Qwen'\n")
            self._write(root / "providers/alibaba/provider.toml", "name = 'Alibaba'\n")
            self._write(
                root / "stats/openrouter/alibaba/models/alibaba/qwen3.8-2.4t-a95b.json",
                json.dumps({"uptime_last_1d": 99.5}) + "\n",
            )
            stale = root / "missing/providers/alibaba/models/stale.toml"
            self._write(stale, detector.GENERATED_MARKER + "\n")
            manual = root / "missing/providers/manual.txt"
            self._write(manual, "keep\n")

            result = detector.scan_repository(root)
            written, errors = detector.write_missing_provider_outputs(root, result)

            provider_output = root / "missing/providers/alibaba/provider.toml"
            model_output = root / "missing/providers/alibaba/models/qwen3.8-2.4t-a95b.toml"
            provider_content = provider_output.read_text(encoding="utf-8")
            model_content = model_output.read_text(encoding="utf-8")
            manual_exists = manual.exists()

        self.assertEqual(written, 2)
        self.assertEqual(errors, [])
        self.assertFalse(stale.exists())
        self.assertTrue(manual_exists)
        self.assertIn(detector.GENERATED_MARKER, provider_content)
        self.assertIn(detector.GENERATED_MARKER, model_content)
        self.assertIn("alibaba/qwen3.8-2.4t-a95b", model_content)
        self.assertEqual(
            tomllib.loads(model_content),
            {"base_model": "alibaba/qwen3.8-2.4t-a95b"},
        )

    def test_writes_synthetic_stats_from_matching_provider_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            self._write(root / "models/alibaba/qwen3.8-2.4t-a95b.toml", "name = 'Qwen'\n")
            self._write(root / "providers/alibaba/provider.toml", "name = 'Alibaba'\n")
            self._write(
                root / "providers/openrouter/models/qwen/qwen3.8-2.4t-a95b.toml",
                'base_model = "alibaba/qwen3.8-2.4t-a95b"\n',
            )
            source = root / "stats/openrouter/alibaba/models/alibaba/qwen3.8-2.4t-a95b.json"
            source.parent.mkdir(parents=True)
            payload = {
                "uptime_last_1d": 99.5,
                "latency_last_30m": {"p50": 1671},
                "throughput_last_30m": {"p50": 44},
            }
            source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            self._write(
                root / "stats/openrouter/deepinfra/models/alibaba/qwen3.8-2.4t-a95b.json",
                json.dumps({"uptime_last_1d": 95}) + "\n",
            )

            result = detector.scan_repository(root)
            written, errors = detector.write_synthetic_missing_stats(root, result)

            destination = root / "stats/_synthetic/openrouter/openrouter/models/qwen/qwen3.8-2.4t-a95b.json"
            written_payload = json.loads(destination.read_text(encoding="utf-8"))

        self.assertEqual(written, 1)
        self.assertEqual(errors, [])
        self.assertEqual(written_payload, payload)

    def test_dry_run_reports_synthetic_files_without_mutating_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            self._write(root / "providers/alibaba/provider.toml", "name = 'Alibaba'\n")
            self._write(
                root / "stats/openrouter/alibaba/models/alibaba/qwen3.8-2.4t-a95b.json",
                json.dumps({"uptime_last_1d": 99.5}) + "\n",
            )
            output = root / "missing/providers/alibaba/models/old.toml"
            self._write(output, detector.GENERATED_MARKER + "\n")

            result = detector.scan_repository(root)
            written, errors = detector.write_missing_provider_outputs(root, result, dry_run=True)
            output_exists = output.exists()
            model_output_exists = (root / "missing/providers/alibaba/models/qwen3.8-2.4t-a95b.toml").exists()

        self.assertEqual(written, 2)
        self.assertEqual(errors, [])
        self.assertTrue(output_exists)
        self.assertFalse(model_output_exists)

    @staticmethod
    def _write(path: pathlib.Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
