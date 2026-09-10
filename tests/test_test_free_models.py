#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "test_free_models.py"


def load_script():
    spec = importlib.util.spec_from_file_location("test_free_models", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tfm = load_script()


def pareto_fixture() -> dict:
    return {
        "stats": {
            "alibaba/qwen3.5-122b-a10b": {
                "providers": {
                    "cortecs": {"qwen3.5-122b-a10b": {"cost": {"input": 0.495, "output": 3.46}}},
                    "zenmux": {"Qwen/Qwen3.5-122B-A10B": {"cost": {"input": 0, "output": 0}}},
                }
            },
            "openai/gpt-5": {
                "providers": {
                    "inferx": {"openai/gpt-5": {"cost": {"input": 0, "output": 0}}},
                    "nan": {"gpt-5": {"cost": {"input": 0, "output": 0}}},
                }
            },
        }
    }


class CollectFreeTriplesTest(unittest.TestCase):
    def test_collects_only_zero_cost_triples_sorted(self):
        triples = tfm.collect_free_triples(pareto_fixture())
        self.assertEqual(
            triples,
            [
                ("alibaba/qwen3.5-122b-a10b", "zenmux", "Qwen/Qwen3.5-122B-A10B"),
                ("openai/gpt-5", "inferx", "openai/gpt-5"),
                ("openai/gpt-5", "nan", "gpt-5"),
            ],
        )

    def test_ignores_malformed_entries(self):
        pareto = {"stats": {"m": {"providers": {"p": {"model": "not-a-dict", "ok": {"cost": {}}}}}}}
        self.assertEqual(tfm.collect_free_triples(pareto), [])


class ProviderConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = __import__("tempfile").TemporaryDirectory()
        self.repo_root = pathlib.Path(self._tmp.name)
        (self.repo_root / "providers" / "zenmux").mkdir(parents=True)
        (self.repo_root / "providers" / "zenmux" / "provider.toml").write_text(
            'name = "ZenMux"\napi = "https://zenmux.ai/api/v1/"\nenv = ["ZENMUX_API_KEY"]\n', encoding="utf-8"
        )
        (self.repo_root / "providers" / "vercel").mkdir()
        (self.repo_root / "providers" / "vercel" / "provider.toml").write_text(
            'name = "Vercel"\nenv = ["AI_GATEWAY_API_KEY"]\n', encoding="utf-8"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_api_from_toml_trailing_slash_stripped(self):
        self.assertEqual(tfm.provider_api_base(self.repo_root, "zenmux"), "https://zenmux.ai/api/v1")

    def test_fallback_api_when_toml_has_no_api(self):
        self.assertEqual(tfm.provider_api_base(self.repo_root, "vercel"), "https://ai-gateway.vercel.sh/v1")

    def test_unknown_provider_has_empty_api(self):
        self.assertEqual(tfm.provider_api_base(self.repo_root, "ghost"), "")

    def test_env_var_from_toml(self):
        self.assertEqual(tfm.provider_env_var(self.repo_root, "zenmux"), "ZENMUX_API_KEY")
        self.assertEqual(tfm.provider_env_var(self.repo_root, "ghost"), "")


class ExtractTextTest(unittest.TestCase):
    def test_chat_completions_shape(self):
        payload = {"choices": [{"message": {"content": " Qwen "}}]}
        self.assertEqual(tfm.extract_text(payload), " Qwen ")

    def test_responses_output_shape(self):
        payload = {"output": [{"content": [{"type": "output_text", "text": "GPT"}]}]}
        self.assertEqual(tfm.extract_text(payload), "GPT")

    def test_responses_output_text_shortcut(self):
        self.assertEqual(tfm.extract_text({"output_text": "hi"}), "hi")

    def test_empty_payload(self):
        self.assertEqual(tfm.extract_text({}), "")
        self.assertEqual(tfm.extract_text({"choices": [{"message": {"content": ""}}]}), "")


class TestTripleProbeOrderTest(unittest.TestCase):
    def _run(self, probe_results):
        calls = []

        def fake_post(url, api_key, payload, timeout_s=tfm.REQUEST_TIMEOUT_S):
            calls.append(url)
            status, body = probe_results[len(calls) - 1]
            return status, json.dumps(body)

        with mock.patch.object(tfm, "post_json", side_effect=fake_post):
            latency = tfm.test_triple("https://x/v1", "key", "model-a")
        return latency, calls

    def test_responses_first_success(self):
        latency, calls = self._run([(200, {"output_text": "model-a"})])
        self.assertEqual(calls, ["https://x/v1/responses"])
        self.assertIsInstance(latency, int)
        self.assertGreaterEqual(latency, 0)

    def test_falls_back_to_chat_completions(self):
        latency, calls = self._run(
            [
                (404, {"error": "no responses endpoint"}),
                (200, {"choices": [{"message": {"content": "model-a"}}]}),
            ]
        )
        self.assertEqual(calls, ["https://x/v1/responses", "https://x/v1/chat/completions"])
        self.assertIsInstance(latency, int)

    def test_both_fail_returns_none(self):
        latency, calls = self._run([(500, {}), (429, {})])
        self.assertIsNone(latency)
        self.assertEqual(len(calls), 2)

    def test_200_with_empty_text_is_not_success(self):
        latency, _ = self._run([(200, {"output": []}), (200, {"choices": []})])
        self.assertIsNone(latency)


class TestFreeModelsTest(unittest.TestCase):
    def test_regenerates_file_shape_and_skips_untestable(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            (repo_root / "providers" / "zenmux").mkdir(parents=True)
            (repo_root / "providers" / "zenmux" / "provider.toml").write_text(
                'api = "https://zenmux.ai/api/v1"\nenv = ["ZENMUX_API_KEY"]\n', encoding="utf-8"
            )
            (repo_root / "providers" / "nan").mkdir()
            (repo_root / "providers" / "nan" / "provider.toml").write_text(
                'api = "https://nan.example/v1"\nenv = ["NAN_API_KEY"]\n', encoding="utf-8"
            )

            latencies = {
                ("https://zenmux.ai/api/v1", "ZM", "Qwen/Qwen3.5-122B-A10B"): 123,
            }

            def fake_tester(api_base, api_key, provider_model):
                return latencies.get((api_base, api_key, provider_model))

            result = tfm.test_free_models(
                repo_root,
                pareto_fixture(),
                env={"ZENMUX_API_KEY": "ZM"},  # NAN_API_KEY missing -> skip
                tester=fake_tester,
            )

        self.assertEqual(
            result,
            {
                "free": {
                    "alibaba/qwen3.5-122b-a10b": {
                        "providers": {"zenmux": {"Qwen/Qwen3.5-122B-A10B": {"latency_ms": 123}}}
                    }
                }
            },
        )

    def test_sorted_output_regardless_of_input_order(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            (repo_root / "providers" / "zenmux").mkdir(parents=True)
            (repo_root / "providers" / "zenmux" / "provider.toml").write_text(
                'api = "https://z/v1"\nenv = ["ZENMUX_API_KEY"]\n', encoding="utf-8"
            )
            (repo_root / "providers" / "inferx").mkdir()
            (repo_root / "providers" / "inferx" / "provider.toml").write_text(
                'api = "https://i/v1"\nenv = ["INFERX_API_KEY"]\n', encoding="utf-8"
            )
            (repo_root / "providers" / "nan").mkdir()
            (repo_root / "providers" / "nan" / "provider.toml").write_text(
                'api = "https://n/v1"\nenv = ["NAN_API_KEY"]\n', encoding="utf-8"
            )

            result = tfm.test_free_models(
                repo_root,
                pareto_fixture(),
                env={"ZENMUX_API_KEY": "k", "INFERX_API_KEY": "k", "NAN_API_KEY": "k"},
                tester=lambda api, key, model: 5,
            )

        labs = list(result["free"].keys())
        self.assertEqual(labs, sorted(labs))
        providers = list(result["free"]["openai/gpt-5"]["providers"].keys())
        self.assertEqual(providers, sorted(providers))


class OutputDeterminismTest(unittest.TestCase):
    def test_main_writes_file(self):
        with __import__("tempfile").TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            pareto_path = repo_root / "pareto.json"
            output_path = repo_root / "tested_models.json"
            pareto_path.write_text(json.dumps(pareto_fixture()), encoding="utf-8")
            with (
                mock.patch.object(tfm, "REPO_ROOT", repo_root),
                mock.patch.object(tfm, "PARETO_PATH", pareto_path),
                mock.patch.object(tfm, "OUTPUT_PATH", output_path),
                mock.patch.object(tfm, "test_free_models", return_value={"free": {}}) as tester,
            ):
                tfm.main()
            self.assertTrue(output_path.exists())
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), {"free": {}})
            tester.assert_called_once()


if __name__ == "__main__":
    unittest.main()
