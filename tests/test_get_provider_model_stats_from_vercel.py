#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / ".github" / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "get_provider_model_stats_from_vercel.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.provider_model_stats import write_outputs, write_synthetic_stats  # noqa: E402


def load_script():
    spec = importlib.util.spec_from_file_location("get_provider_model_stats_from_vercel", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_provider_model_stats = load_script()


class VercelStatsTest(unittest.TestCase):
    def test_debug_option_is_disabled_by_default_and_can_be_enabled(self):
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH)]):
            self.assertFalse(get_provider_model_stats.parse_args().debug)
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH), "--debug"]):
            self.assertTrue(get_provider_model_stats.parse_args().debug)

    def test_uses_shared_ai_provider_mapping_function(self):
        from lib.provider_model_stats import query_provider_mappings

        self.assertIs(get_provider_model_stats.query_provider_mappings, query_provider_mappings)

    def test_extracts_vercel_metrics_without_renaming_or_filling(self):
        endpoint = {
            "provider_name": "alibaba",
            "uptime_last_15m": 100,
            "uptime_last_1h": 99.5,
            "uptime_last_1d": 98.75,
            "latency_last_1h": {"p50": 919, "p95": 1946.2},
            "throughput_last_1h": {"p50": 161, "p95": 189.7},
        }

        self.assertEqual(
            get_provider_model_stats.extract_stats(endpoint),
            {
                "uptime_last_15m": 100,
                "uptime_last_1h": 99.5,
                "uptime_last_1d": 98.75,
                "latency_last_1h": {"p50": 919, "p95": 1946.2},
                "throughput_last_1h": {"p50": 161, "p95": 189.7},
            },
        )

    def test_does_not_fill_missing_vercel_metrics(self):
        endpoint = {
            "provider_name": "alibaba",
            "uptime_last_1h": 99.5,
            "latency_last_1h": {"p50": 919},
        }

        self.assertEqual(
            get_provider_model_stats.extract_stats(endpoint),
            {
                "uptime_last_1h": 99.5,
                "latency_last_1h": {"p50": 919},
            },
        )

    def test_parse_vercel_response_requires_endpoints_list(self):
        with self.assertRaises(TypeError):
            get_provider_model_stats.parse_vercel_response(
                {"data": {"id": "alibaba/qwen3.5-plus"}}, "alibaba/qwen3.5-plus"
            )

    def test_writes_stats_as_json_without_schema_conversion(self):
        outputs = {
            "alibaba/models/alibaba/qwen3.5-plus.json": {
                "uptime_last_1h": 100,
                "latency_last_1h": {"p50": 919, "p95": 1946.2},
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = pathlib.Path(temporary_directory)
            written = write_outputs(outputs, output_dir, dry_run=False)
            destination = output_dir / "alibaba/models/alibaba/qwen3.5-plus.json"

            self.assertEqual(written, 1)
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                outputs[destination.relative_to(output_dir).as_posix()],
            )

    def test_writes_vercel_synthetic_stats_to_requested_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            providers_dir = root / "providers"
            routers_dir = root / "routers"
            stats_dir = root / "stats" / "vercel"
            synthetic_dir = root / "stats" / "_synthetic" / "vercel"
            routers_dir.mkdir()
            (routers_dir / "vercel.toml").write_text(
                "is_router = true\nrouter_providers_from_models_dir_struct = true\n",
                encoding="utf-8",
            )
            source = stats_dir / "alibaba" / "models" / "alibaba" / "qwen3.5-plus.json"
            source.parent.mkdir(parents=True)
            payload = {
                "uptime_last_1h": 100,
                "latency_last_1h": {"p50": 919, "p95": 1946.2},
                "throughput_last_1h": {"p50": 161, "p95": 189.7},
            }
            source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            model = providers_dir / "vercel" / "models" / "alibaba" / "qwen3.5-plus.toml"
            model.parent.mkdir(parents=True)
            model.write_text('base_model = "alibaba/qwen3.5-plus"\n', encoding="utf-8")

            written, errors, collisions = write_synthetic_stats(
                providers_dir=providers_dir,
                routers_dir=routers_dir,
                stats_dir=stats_dir,
                synthetic_dir=synthetic_dir,
                dry_run=False,
            )

            destination = synthetic_dir / "vercel" / "models" / "alibaba" / "qwen3.5-plus.json"
            self.assertEqual(written, 1)
            self.assertEqual(errors, [])
            self.assertEqual(collisions, [])
            self.assertEqual(destination.read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
