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
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "test_paid_models.py"


def load_script():
    spec = importlib.util.spec_from_file_location("test_paid_models", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tpm = load_script()


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
                    "nan": {"gpt-5": {"cost": {"input": 0.5, "output": 1.5}}},
                }
            },
        }
    }


class CollectPaidTriplesTest(unittest.TestCase):
    def test_collects_only_nonzero_cost_triples_sorted(self):
        triples = tpm.collect_paid_triples(pareto_fixture())
        self.assertEqual(
            triples,
            [
                ("alibaba/qwen3.5-122b-a10b", "cortecs", "qwen3.5-122b-a10b"),
                ("openai/gpt-5", "nan", "gpt-5"),
            ],
        )

    def test_zero_cost_triples_are_excluded(self):
        triples = tpm.collect_paid_triples(pareto_fixture())
        self.assertNotIn(("alibaba/qwen3.5-122b-a10b", "zenmux", "Qwen/Qwen3.5-122B-A10B"), triples)
        self.assertNotIn(("openai/gpt-5", "inferx", "openai/gpt-5"), triples)

    def test_missing_cost_counts_as_skipped(self):
        pareto = {"stats": {"m": {"providers": {"p": {"model": {"cost": {}}}}}}}
        # empty cost dict: input/output default to 0 -> not provably paid
        self.assertEqual(tpm.collect_paid_triples(pareto), [])

    def test_ignores_malformed_entries(self):
        pareto = {"stats": {"m": {"providers": {"p": {"model": "not-a-dict", "ok": {"cost": {"input": 1}}}}}}}
        # "model" is a non-dict entry and must be ignored; "ok" has a non-zero
        # cost and counts as paid.
        self.assertEqual(tpm.collect_paid_triples(pareto), [("m", "p", "ok")])


class ProviderConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
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
        self.assertEqual(tpm.provider_api_base(self.repo_root, "zenmux"), "https://zenmux.ai/api/v1")

    def test_fallback_api_when_toml_has_no_api(self):
        self.assertEqual(tpm.provider_api_base(self.repo_root, "vercel"), "https://ai-gateway.vercel.sh/v1")

    def test_unknown_provider_has_empty_api(self):
        self.assertEqual(tpm.provider_api_base(self.repo_root, "ghost"), "")

    def test_env_var_from_toml(self):
        self.assertEqual(tpm.provider_env_var(self.repo_root, "zenmux"), "ZENMUX_API_KEY")
        self.assertEqual(tpm.provider_env_var(self.repo_root, "ghost"), "")


class TestTripleProbeOrderTest(unittest.TestCase):
    def _run(self, probe_results):
        calls = []

        def fake_post(url, api_key, payload, timeout_s=tpm.REQUEST_TIMEOUT_S):
            calls.append((url, payload))
            status, body = probe_results[len(calls) - 1]
            return status, json.dumps(body)

        with mock.patch.object(tpm, "post_json", side_effect=fake_post):
            outcome = tpm.test_triple("https://x/v1", "key", "model-a")
        return outcome, calls

    def test_dot_input_payload_for_responses(self):
        outcome, calls = self._run([(402, {})])
        (url, payload) = calls[0]
        self.assertEqual(url, "https://x/v1/responses")
        self.assertEqual(payload, {"model": "model-a", "input": "."})

    def test_dot_message_payload_for_chat_completions(self):
        outcome, calls = self._run([(404, {}), (402, {})])
        (url, payload) = calls[1]
        self.assertEqual(url, "https://x/v1/chat/completions")
        self.assertEqual(
            payload,
            {"model": "model-a", "messages": [{"role": "user", "content": "."}]},
        )

    def test_402_on_responses_short_circuits_chat_completions(self):
        outcome, calls = self._run([(402, {"error": "no credit"})])
        self.assertEqual(len(calls), 1)
        latency_ms, endpoint_type = outcome
        self.assertIsInstance(latency_ms, int)
        self.assertGreaterEqual(latency_ms, 0)
        self.assertEqual(endpoint_type, "responses")

    def test_403_counts_as_existing(self):
        outcome, calls = self._run([(403, {"error": "insufficient balance"})])
        self.assertEqual(len(calls), 1)
        self.assertEqual(outcome[1], "responses")

    def test_falls_back_to_chat_completions_on_404(self):
        outcome, calls = self._run(
            [
                (404, {"error": "no responses endpoint"}),
                (402, {"error": "no credit"}),
            ]
        )
        self.assertEqual(calls[0][0], "https://x/v1/responses")
        self.assertEqual(calls[1][0], "https://x/v1/chat/completions")
        self.assertEqual(outcome[1], "chat/completions")

    def test_200_counts_as_missing(self):
        outcome, calls = self._run([(200, {"output": []}), (200, {"choices": []})])
        self.assertIsNone(outcome)
        self.assertEqual(len(calls), 2)

    def test_400_counts_as_missing(self):
        outcome, calls = self._run([(400, {"error": "bad request"}), (400, {"error": "bad request"})])
        self.assertIsNone(outcome)
        self.assertEqual(len(calls), 2)

    def test_both_404_returns_none(self):
        outcome, calls = self._run([(404, {}), (404, {})])
        self.assertIsNone(outcome)
        self.assertEqual(len(calls), 2)

    def test_probe_error_counts_as_missing(self):
        with mock.patch.object(tpm, "post_json", side_effect=OSError("boom")):
            outcome = tpm.test_triple("https://x/v1", "key", "model-a")
        self.assertIsNone(outcome)

    def test_custom_exists_codes(self):
        # default codes: 429 does not prove existence
        outcome, calls = self._run([(404, {}), (429, {"error": "rate limited"})])
        self.assertIsNone(outcome)
        self.assertEqual(len(calls), 2)
        # custom codes: 429 proves existence
        with mock.patch.object(tpm, "post_json", return_value=(429, "{}")):
            outcome = tpm.test_triple("https://x/v1", "key", "model-a", exists_codes=(429,))
        self.assertEqual(outcome[1], "responses")


class TestPaidModelsTest(unittest.TestCase):
    def _prepare_repo(self, repo_root: pathlib.Path) -> None:
        (repo_root / "providers" / "zenmux").mkdir(parents=True)
        (repo_root / "providers" / "zenmux" / "provider.toml").write_text(
            'api = "https://zenmux.ai/api/v1"\nenv = ["ZENMUX_API_KEY"]\n', encoding="utf-8"
        )
        (repo_root / "providers" / "nan").mkdir()
        (repo_root / "providers" / "nan" / "provider.toml").write_text(
            'api = "https://nan.example/v1"\nenv = ["NAN_API_KEY"]\n', encoding="utf-8"
        )

    def test_rebuilds_paid_branch_and_skips_untestable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            self._prepare_repo(repo_root)

            outcomes = {
                ("https://nan.example/v1", "N", "gpt-5"): (123, "responses"),
            }

            def fake_tester(api_base, api_key, provider_model):
                return outcomes.get((api_base, api_key, provider_model))

            result = tpm.test_paid_models(
                repo_root,
                pareto_fixture(),
                env={tpm.BULK_KEYS_ENV: json.dumps({"NAN_API_KEY": "N"})},  # no cortecs key -> skip
                tester=fake_tester,
            )

        self.assertEqual(
            result,
            {
                "paid": {
                    "openai/gpt-5": {"providers": {"nan": {"gpt-5": {"latency_ms": 123, "endpoint_type": "responses"}}}}
                }
            },
        )

    def test_excluded_provider_triples_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            self._prepare_repo(repo_root)
            (repo_root / "providers" / "cortecs").mkdir()
            (repo_root / "providers" / "cortecs" / "provider.toml").write_text(
                'api = "https://c/v1"\nenv = ["CORTECS_API_KEY"]\n', encoding="utf-8"
            )

            calls = []

            def fake_tester(api_base, api_key, provider_model):
                calls.append((api_base, api_key, provider_model))
                return (1, "responses")

            with mock.patch.dict(tpm.EXCLUDED_PROVIDERS, {"cortecs": "blocked"}):
                result = tpm.test_paid_models(
                    repo_root,
                    pareto_fixture(),
                    env={tpm.BULK_KEYS_ENV: json.dumps({"CORTECS_API_KEY": "k", "NAN_API_KEY": "N"})},
                    tester=fake_tester,
                )

        tested = {(api, key, model) for api, key, model in calls}
        # cortecs triple must not reach the tester
        self.assertNotIn(("https://c/v1", "k", "qwen3.5-122b-a10b"), tested)
        # nan triple was probed and recorded
        self.assertIn(("https://nan.example/v1", "N", "gpt-5"), tested)
        self.assertEqual(
            result["paid"],
            {"openai/gpt-5": {"providers": {"nan": {"gpt-5": {"latency_ms": 1, "endpoint_type": "responses"}}}}},
        )

    def test_models_determinant_uses_catalogue(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            self._prepare_repo(repo_root)
            (repo_root / "providers" / "kilo").mkdir()
            (repo_root / "providers" / "kilo" / "provider.toml").write_text(
                'api = "https://kilo.example/v1"\nenv = ["KILO_API_KEY"]\n', encoding="utf-8"
            )

            probes = []

            def fake_probe(api_base, api_key, provider_model, exists_codes=(402, 403)):
                probes.append(provider_model)
                return None

            catalogue = {"gpt-5"}

            def fake_fetch(api_base, api_key):
                return catalogue

            pareto = {
                "stats": {
                    "openai/gpt-5": {
                        "providers": {
                            "kilo": {"gpt-5": {"cost": {"input": 0.5, "output": 1.5}}},
                            "kilo-missing": {"ghost": {"cost": {"input": 0.5, "output": 1.5}}},
                        }
                    }
                }
            }
            with (
                mock.patch.dict(tpm.PROVIDER_DETERMINANTS, {"kilo": "models", "kilo-missing": "models"}),
                mock.patch.object(tpm, "fetch_model_ids", side_effect=fake_fetch),
                mock.patch.object(tpm, "test_triple", side_effect=fake_probe),
            ):
                result = tpm.test_paid_models(
                    repo_root,
                    pareto,
                    env={tpm.BULK_KEYS_ENV: json.dumps({"KILO_API_KEY": "k"})},
                )

        # listed model recorded via the catalogue without any inference probe
        self.assertEqual(
            result["paid"],
            {"openai/gpt-5": {"providers": {"kilo": {"gpt-5": {"latency_ms": 0, "endpoint_type": "models"}}}}},
        )
        self.assertEqual(probes, [])

    def test_preserves_other_branches_of_existing_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            self._prepare_repo(repo_root)
            existing = {
                "free": {"openai/gpt-5": {"providers": {"inferx": {"openai/gpt-5": {"latency_ms": 7}}}}},
                "manual": {"note": "kept as-is"},
            }
            result = tpm.test_paid_models(
                repo_root,
                pareto_fixture(),
                existing=existing,
                env={tpm.BULK_KEYS_ENV: json.dumps({"ZENMUX_API_KEY": "z", "NAN_API_KEY": "N"})},
                tester=lambda api, key, model: (5, "chat/completions"),
            )
        self.assertIn("free", result)
        self.assertIn("manual", result)
        self.assertEqual(result["free"], existing["free"])
        self.assertEqual(result["manual"], existing["manual"])
        paid_labs = list(result["paid"].keys())
        self.assertEqual(paid_labs, sorted(paid_labs))

    def test_sorted_output_regardless_of_input_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            self._prepare_repo(repo_root)
            (repo_root / "providers" / "cortecs").mkdir()
            (repo_root / "providers" / "cortecs" / "provider.toml").write_text(
                'api = "https://c/v1"\nenv = ["CORTECS_API_KEY"]\n', encoding="utf-8"
            )

            result = tpm.test_paid_models(
                repo_root,
                pareto_fixture(),
                env={
                    tpm.BULK_KEYS_ENV: json.dumps({"ZENMUX_API_KEY": "k", "CORTECS_API_KEY": "k", "NAN_API_KEY": "k"})
                },
                tester=lambda api, key, model: (5, "responses"),
            )

        labs = list(result["paid"].keys())
        self.assertEqual(labs, sorted(labs))
        providers = list(result["paid"]["openai/gpt-5"]["providers"].keys())
        self.assertEqual(providers, sorted(providers))

    def test_parallel_execution_processes_all_triples(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            self._prepare_repo(repo_root)
            (repo_root / "providers" / "cortecs").mkdir()
            (repo_root / "providers" / "cortecs" / "provider.toml").write_text(
                'api = "https://c/v1"\nenv = ["CORTECS_API_KEY"]\n', encoding="utf-8"
            )

            result = tpm.test_paid_models(
                repo_root,
                pareto_fixture(),
                env={
                    tpm.BULK_KEYS_ENV: json.dumps({"CORTECS_API_KEY": "k", "NAN_API_KEY": "k"}),
                },
                tester=lambda api, key, model: (1, "responses"),
                max_workers=4,
            )

        tested = sum(len(models) for info in result["paid"].values() for models in info["providers"].values())
        self.assertEqual(tested, 2)


class ResolveApiKeysTest(unittest.TestCase):
    def test_parses_bulk_secret(self):
        env = {tpm.BULK_KEYS_ENV: json.dumps({"ZENMUX_API_KEY": " a ", "NAN_API_KEY": " n "})}
        self.assertEqual(tpm.resolve_api_keys(env), {"ZENMUX_API_KEY": "a", "NAN_API_KEY": "n"})

    def test_missing_bulk_env_is_fatal(self):
        with self.assertRaises(RuntimeError):
            tpm.resolve_api_keys({})

    def test_invalid_json_is_fatal(self):
        with self.assertRaises(RuntimeError):
            tpm.resolve_api_keys({tpm.BULK_KEYS_ENV: "{not json"})

    def test_non_object_json_is_fatal(self):
        with self.assertRaises(RuntimeError):
            tpm.resolve_api_keys({tpm.BULK_KEYS_ENV: "[]"})

    def test_blank_values_are_dropped(self):
        tpm.resolve_api_keys({tpm.BULK_KEYS_ENV: json.dumps({"A_API_KEY": "", "B_API_KEY": "  "})})
        self.assertEqual(tpm.resolve_api_keys({tpm.BULK_KEYS_ENV: "{}"}), {})
        self.assertEqual(tpm.resolve_api_keys({tpm.BULK_KEYS_ENV: json.dumps({"A_API_KEY": "x"})}), {"A_API_KEY": "x"})

    def test_no_fallback_to_individual_env_vars(self):
        with self.assertRaises(RuntimeError):
            tpm.test_paid_models(
                pathlib.Path("."),
                pareto_fixture(),
                env={"NAN_API_KEY": "individual-env-value"},
                tester=lambda api, key, model: (1, "responses"),
            )


class OutputLoadingTest(unittest.TestCase):
    def test_load_output_missing_file_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(tpm.load_output(pathlib.Path(tmp) / "missing.json"), {})

    def test_load_output_invalid_json_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(tpm.load_output(path), {})

    def test_load_output_non_object_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "array.json"
            path.write_text("[]", encoding="utf-8")
            self.assertEqual(tpm.load_output(path), {})

    def test_main_merges_into_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            pareto_path = repo_root / "pareto.json"
            output_path = repo_root / "tested_models.json"
            pareto_path.write_text(json.dumps(pareto_fixture()), encoding="utf-8")
            output_path.write_text(
                json.dumps({"free": {"keep": True}}),
                encoding="utf-8",
            )
            paid_result = {
                "paid": {
                    "openai/gpt-5": {"providers": {"nan": {"gpt-5": {"latency_ms": 1, "endpoint_type": "responses"}}}}
                }
            }
            with (
                mock.patch.object(tpm, "REPO_ROOT", repo_root),
                mock.patch.object(tpm, "PARETO_PATH", pareto_path),
                mock.patch.object(tpm, "OUTPUT_PATH", output_path),
                mock.patch.object(
                    tpm, "test_paid_models", return_value={"free": {"keep": True}, **paid_result}
                ) as tester,
            ):
                tpm.main()
            merged = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(merged, {"free": {"keep": True}, **paid_result})
            # existing payload was passed into test_paid_models
            self.assertEqual(tester.call_args.kwargs["existing"], {"free": {"keep": True}})


if __name__ == "__main__":
    unittest.main()
