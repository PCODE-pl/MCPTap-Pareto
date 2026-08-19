#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "compute_average_provider_model_stats.py"


def load_script():
    spec = importlib.util.spec_from_file_location("compute_average_provider_model_stats", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compute_average_stats = load_script()


class ComputeAverageProviderModelStatsTest(unittest.TestCase):
    def test_combines_direct_and_synthetic_sources_for_nested_model_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            model = root / "providers" / "acme" / "models" / "lab" / "nested" / "model.toml"
            model.parent.mkdir(parents=True)
            model.write_text('base_model = "lab/model"\n', encoding="utf-8")

            openrouter = root / "stats" / "openrouter" / "acme" / "models" / "lab" / "nested" / "model.json"
            openrouter.parent.mkdir(parents=True)
            openrouter.write_text(
                json.dumps(
                    {
                        "uptime_last_5m": 90,
                        "uptime_last_30m": 91,
                        "uptime_last_1d": 80,
                        "latency_last_30m": {"p50": 100, "p95": 200},
                        "throughput_last_30m": {"p50": 20},
                    }
                ),
                encoding="utf-8",
            )

            vercel = root / "stats" / "_synthetic" / "vercel" / "acme" / "models" / "lab" / "nested" / "model.json"
            vercel.parent.mkdir(parents=True)
            vercel.write_text(
                json.dumps(
                    {
                        "uptime_last_15m": 92,
                        "uptime_last_1h": 93,
                        "uptime_last_1d": 100,
                        "latency_last_1h": {"p50": 120, "p95": 300},
                        "throughput_last_1h": {"p50": 30},
                    }
                ),
                encoding="utf-8",
            )

            written, errors = compute_average_stats.compute_stats(root, dry_run=False)

            destination = root / "stats" / "_average" / "acme" / "models" / "lab" / "nested" / "model.json"
            self.assertEqual(written, 1)
            self.assertEqual(errors, [])
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {
                    "uptime_last_5m": 90,
                    "uptime_last_15m": 92,
                    "uptime_last_30m": 91,
                    "uptime_last_1h": 93,
                    "uptime_last_1d": 90.0,
                    "latency_last_30m": {"p50": 100},
                    "latency_last_1h": {"p50": 120},
                    "throughput_last_30m": {"p50": 20},
                    "throughput_last_1h": {"p50": 30},
                },
            )

    def test_prefers_direct_file_and_omits_missing_or_null_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            model = root / "providers" / "acme" / "models" / "model.toml"
            model.parent.mkdir(parents=True)
            model.write_text("", encoding="utf-8")

            direct = root / "stats" / "openrouter" / "acme" / "models" / "model.json"
            direct.parent.mkdir(parents=True)
            direct.write_text(
                json.dumps(
                    {
                        "uptime_last_5m": None,
                        "uptime_last_30m": 97,
                        "latency_last_30m": {"p50": None},
                    }
                ),
                encoding="utf-8",
            )
            synthetic = root / "stats" / "_synthetic" / "openrouter" / "acme" / "models" / "model.json"
            synthetic.parent.mkdir(parents=True)
            synthetic.write_text(json.dumps({"uptime_last_5m": 1, "uptime_last_1d": 2}), encoding="utf-8")

            written, errors = compute_average_stats.compute_stats(root, dry_run=False)

            destination = root / "stats" / "_average" / "acme" / "models" / "model.json"
            self.assertEqual(written, 1)
            self.assertEqual(errors, [])
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"uptime_last_30m": 97})

    def test_ignores_uptime_values_below_configured_minimum(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            model = root / "providers" / "acme" / "models" / "model.toml"
            model.parent.mkdir(parents=True)
            model.write_text("", encoding="utf-8")

            openrouter = root / "stats" / "openrouter" / "acme" / "models" / "model.json"
            openrouter.parent.mkdir(parents=True)
            openrouter.write_text(
                json.dumps(
                    {
                        "uptime_last_5m": 29,
                        "uptime_last_30m": 30,
                        "uptime_last_1d": 20,
                    }
                ),
                encoding="utf-8",
            )
            vercel = root / "stats" / "vercel" / "acme" / "models" / "model.json"
            vercel.parent.mkdir(parents=True)
            vercel.write_text(
                json.dumps(
                    {
                        "uptime_last_15m": 29,
                        "uptime_last_1h": 30,
                        "uptime_last_1d": 40,
                    }
                ),
                encoding="utf-8",
            )

            written, errors = compute_average_stats.compute_stats(root, dry_run=False)

            destination = root / "stats" / "_average" / "acme" / "models" / "model.json"
            self.assertEqual(written, 1)
            self.assertEqual(errors, [])
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {
                    "uptime_last_30m": 30,
                    "uptime_last_1h": 30,
                    "uptime_last_1d": 40,
                },
            )

    def test_averages_llmgateway_metrics_and_omits_log_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            model = root / "providers" / "acme" / "models" / "model.toml"
            model.parent.mkdir(parents=True)
            model.write_text("", encoding="utf-8")

            source = root / "stats" / "llmgateway" / "acme" / "models" / "model.json"
            source.parent.mkdir(parents=True)
            source.write_text(
                json.dumps(
                    {
                        "logsCount": 120,
                        "avgTimeToFirstToken": 450,
                        "tokensPerSecond": 80,
                        "uptime": 99.5,
                    }
                ),
                encoding="utf-8",
            )

            written, errors = compute_average_stats.compute_stats(root, dry_run=False)

            destination = root / "stats" / "_average" / "acme" / "models" / "model.json"
            self.assertEqual(written, 1)
            self.assertEqual(errors, [])
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {
                    "avgTimeToFirstToken": 450,
                    "tokensPerSecond": 80,
                    "uptime": 99.5,
                },
            )

    def test_does_not_write_model_without_any_source_stats(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            model = root / "providers" / "acme" / "models" / "model.toml"
            model.parent.mkdir(parents=True)
            model.write_text("", encoding="utf-8")

            written, errors = compute_average_stats.compute_stats(root, dry_run=False)

            self.assertEqual(written, 0)
            self.assertEqual(errors, [])
            self.assertFalse((root / "stats" / "_average").exists())


if __name__ == "__main__":
    unittest.main()
