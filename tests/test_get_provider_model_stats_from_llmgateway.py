#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "get_provider_model_stats_from_llmgateway.py"


def load_script():
    spec = importlib.util.spec_from_file_location("get_provider_model_stats_from_llmgateway", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_provider_model_stats = load_script()


class LlmGatewayStatsTest(unittest.TestCase):
    def test_parse_public_stats_returns_unique_model_ids(self):
        payload = {
            "models": [
                {"modelId": "qwen3.7-max"},
                {"modelId": "deepseek-v4-pro"},
                {"modelId": "qwen3.7-max"},
            ]
        }

        self.assertEqual(
            get_provider_model_stats.parse_public_stats_response(payload),
            ["deepseek-v4-pro", "qwen3.7-max"],
        )

    def test_parse_public_stats_requires_models_list(self):
        with self.assertRaises(TypeError):
            get_provider_model_stats.parse_public_stats_response({"data": []})

    def test_fetch_public_model_ids_uses_long_backoff_for_public_index(self):
        with mock.patch.object(
            get_provider_model_stats,
            "fetch_json",
            return_value={"models": []},
        ) as fetch_json:
            result = get_provider_model_stats.fetch_public_model_ids("https://internal.example")

        self.assertEqual(result, [])
        fetch_json.assert_called_once_with(
            "https://internal.example/public/models/stats?window=24h",
            headers={"Accept": "application/json"},
            error_context="LLM Gateway public model stats request failed",
            retries=4,
            retry_delay=5,
        )

    def test_parse_benchmarks_maps_provider_id_to_shared_provider_name(self):
        payload = {
            "providers": [
                {
                    "providerId": "alibaba",
                    "logsCount": 120,
                    "avgTimeToFirstToken": 450,
                    "tokensPerSecond": 80,
                    "uptime": 99.5,
                }
            ]
        }

        self.assertEqual(
            get_provider_model_stats.parse_benchmarks_response(payload, "qwen3.7-max"),
            [
                {
                    "provider_name": "alibaba",
                    "logsCount": 120,
                    "avgTimeToFirstToken": 450,
                    "tokensPerSecond": 80,
                    "uptime": 99.5,
                }
            ],
        )

    def test_parse_benchmarks_requires_providers_list(self):
        with self.assertRaises(TypeError):
            get_provider_model_stats.parse_benchmarks_response({"modelId": "qwen3.7-max"}, "qwen3.7-max")

    def test_extract_stats_omits_missing_and_null_values(self):
        endpoint = {
            "provider_name": "alibaba",
            "logsCount": None,
            "avgTimeToFirstToken": 450,
            "tokensPerSecond": None,
            "uptime": 99.5,
        }

        self.assertEqual(
            get_provider_model_stats.extract_stats(endpoint),
            {"avgTimeToFirstToken": 450, "uptime": 99.5},
        )

    def test_fetch_model_endpoints_url_encodes_model_id_as_one_path_segment(self):
        with mock.patch.object(
            get_provider_model_stats,
            "fetch_json",
            return_value={"providers": []},
        ) as fetch_json:
            result = get_provider_model_stats.fetch_model_endpoints("https://internal.example", "model/name")

        self.assertEqual(result, [])
        fetch_json.assert_called_once_with(
            "https://internal.example/internal/models/model%2Fname/benchmarks",
            headers={"Accept": "application/json"},
            error_context="LLM Gateway benchmark request failed for model/name",
            retries=2,
            retry_delay=1,
        )


if __name__ == "__main__":
    unittest.main()
