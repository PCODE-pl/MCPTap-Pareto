#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "get_provider_model_stats_from_vercel.py"


def load_script():
    spec = importlib.util.spec_from_file_location("get_provider_model_stats_from_vercel", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_provider_model_stats = load_script()


class VercelStatsTest(unittest.TestCase):
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

    def test_builds_one_raw_stats_file_per_vercel_endpoint(self):
        models = {
            "alibaba/qwen3.5-plus": [
                {"path": "providers/vercel/models/alibaba/qwen3.5-plus.toml", "model_id": "alibaba/qwen3.5-plus"}
            ]
        }
        endpoints = {
            "alibaba/qwen3.5-plus": [
                {
                    "provider_name": "alibaba",
                    "uptime_last_15m": 100,
                    "uptime_last_1h": 100,
                    "uptime_last_1d": 99.5,
                    "latency_last_1h": {"p50": 919, "p95": 1946.2},
                    "throughput_last_1h": {"p50": 161, "p95": 189.7},
                },
                {
                    "provider_name": "vertex",
                    "uptime_last_15m": 99.2,
                    "uptime_last_1h": 99.7,
                    "uptime_last_1d": 99.9,
                    "latency_last_1h": {"p50": 800, "p95": 1800},
                    "throughput_last_1h": {"p50": 150, "p95": 180},
                },
            ]
        }

        outputs, collisions, unmatched = get_provider_model_stats.build_outputs(models, endpoints)

        self.assertEqual(collisions, [])
        self.assertEqual(unmatched, [])
        self.assertEqual(
            outputs["alibaba/models/alibaba/qwen3.5-plus.json"],
            get_provider_model_stats.extract_stats(endpoints["alibaba/qwen3.5-plus"][0]),
        )
        self.assertEqual(
            outputs["vertex/models/alibaba/qwen3.5-plus.json"],
            get_provider_model_stats.extract_stats(endpoints["alibaba/qwen3.5-plus"][1]),
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
            written = get_provider_model_stats.write_outputs(outputs, output_dir, dry_run=False)
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

            written, errors, collisions = get_provider_model_stats.write_synthetic_stats(
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
