#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
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


class ExtractSseTextTest(unittest.TestCase):
    def test_responses_delta_events(self):
        raw = (
            "event: response.output_text.delta\n"
            + "data: "
            + json.dumps({"type": "response.output_text.delta", "delta": "Mus"})
            + "\n"
            + "data: "
            + json.dumps({"type": "response.output_text.delta", "delta": "e"})
            + "\n\n"
        )
        self.assertEqual(tfm.extract_sse_text(raw), "Muse")

    def test_chat_chunk_choice_deltas(self):
        raw = (
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "Qw"}}]})
            + "\n"
            + "data: "
            + json.dumps({"choices": [{"delta": {"content": "en"}}]})
            + "\n"
            + "data: [DONE]\n\n"
        )
        self.assertEqual(tfm.extract_sse_text(raw), "Qwen")

    def test_non_deltas_ignored(self):
        raw = "data: " + json.dumps({"type": "response.created"}) + "\n\n"
        self.assertEqual(tfm.extract_sse_text(raw), "")
        self.assertEqual(tfm.extract_sse_text("not SSE at all\n"), "")


class TestTripleProbeOrderTest(unittest.TestCase):
    def _run(self, probe_results):
        calls = []

        def fake_post(url, api_key, payload, timeout_s=tfm.REQUEST_TIMEOUT_S):
            calls.append(url)
            status, body = probe_results[len(calls) - 1]
            return status, json.dumps(body)

        with mock.patch.object(tfm, "post_json", side_effect=fake_post):
            outcome = tfm.test_triple("https://x/v1", "key", "model-a")
        return outcome, calls

    def test_responses_success_short_circuits_chat_completions(self):
        outcome, calls = self._run([(200, {"output_text": "model-a"})])
        self.assertEqual(calls, ["https://x/v1/responses"])
        latency_ms, endpoint_type = outcome
        self.assertIsInstance(latency_ms, int)
        self.assertGreaterEqual(latency_ms, 0)
        self.assertEqual(endpoint_type, "responses")

    def test_falls_back_to_chat_completions(self):
        outcome, calls = self._run(
            [
                (404, {"error": "no responses endpoint"}),
                (200, {"choices": [{"message": {"content": "model-a"}}]}),
            ]
        )
        self.assertEqual(calls, ["https://x/v1/responses", "https://x/v1/chat/completions"])
        self.assertEqual(outcome[1], "chat/completions")

    def test_both_fail_returns_none(self):
        outcome, calls = self._run([(500, {}), (429, {})])
        self.assertIsNone(outcome)
        self.assertEqual(len(calls), 2)

    def test_200_with_empty_text_is_not_success(self):
        outcome, _ = self._run([(200, {"output": []}), (200, {"choices": []})])
        self.assertIsNone(outcome)


class TestFreeModelsTest(unittest.TestCase):
    def _prepare_repo(self, repo_root: pathlib.Path) -> None:
        (repo_root / "providers" / "zenmux").mkdir(parents=True)
        (repo_root / "providers" / "zenmux" / "provider.toml").write_text(
            'api = "https://zenmux.ai/api/v1"\nenv = ["ZENMUX_API_KEY"]\n', encoding="utf-8"
        )
        (repo_root / "providers" / "nan").mkdir()
        (repo_root / "providers" / "nan" / "provider.toml").write_text(
            'api = "https://nan.example/v1"\nenv = ["NAN_API_KEY"]\n', encoding="utf-8"
        )

    def test_regenerates_free_branch_and_skips_untestable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            self._prepare_repo(repo_root)

            outcomes = {
                ("https://zenmux.ai/api/v1", "ZM", "Qwen/Qwen3.5-122B-A10B"): (123, "responses"),
            }

            def fake_tester(api_base, api_key, provider_model):
                return outcomes.get((api_base, api_key, provider_model))

            result = tfm.test_free_models(
                repo_root,
                pareto_fixture(),
                env={tfm.BULK_KEYS_ENV: json.dumps({"ZENMUX_API_KEY": "ZM"})},  # NAN_API_KEY missing -> skip
                tester=fake_tester,
            )

        self.assertEqual(
            result,
            {
                "free": {
                    "alibaba/qwen3.5-122b-a10b": {
                        "providers": {
                            "zenmux": {"Qwen/Qwen3.5-122B-A10B": {"latency_ms": 123, "endpoint_type": "responses"}}
                        }
                    }
                }
            },
        )

    def test_preserves_other_branches_of_existing_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            self._prepare_repo(repo_root)
            existing = {
                "paid": {"openai/gpt-5": {"providers": {"inferx": {"openai/gpt-5": {"latency_ms": 7}}}}},
                "manual": {"note": "kept as-is"},
            }
            result = tfm.test_free_models(
                repo_root,
                pareto_fixture(),
                existing=existing,
                env={tfm.BULK_KEYS_ENV: json.dumps({"ZENMUX_API_KEY": "ZM", "NAN_API_KEY": "N"})},
                tester=lambda api, key, model: (5, "chat/completions"),
            )
        self.assertIn("paid", result)
        self.assertIn("manual", result)
        self.assertEqual(result["paid"], existing["paid"])
        self.assertEqual(result["manual"], existing["manual"])
        free_labs = list(result["free"].keys())
        self.assertEqual(free_labs, sorted(free_labs))

    def test_sorted_output_regardless_of_input_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            self._prepare_repo(repo_root)
            (repo_root / "providers" / "inferx").mkdir()
            (repo_root / "providers" / "inferx" / "provider.toml").write_text(
                'api = "https://i/v1"\nenv = ["INFERX_API_KEY"]\n', encoding="utf-8"
            )

            result = tfm.test_free_models(
                repo_root,
                pareto_fixture(),
                env={tfm.BULK_KEYS_ENV: json.dumps({"ZENMUX_API_KEY": "k", "INFERX_API_KEY": "k", "NAN_API_KEY": "k"})},
                tester=lambda api, key, model: (5, "responses"),
            )

        labs = list(result["free"].keys())
        self.assertEqual(labs, sorted(labs))
        providers = list(result["free"]["openai/gpt-5"]["providers"].keys())
        self.assertEqual(providers, sorted(providers))


class OpencodeSessionHeaderTest(unittest.TestCase):
    def _prepare_opencode_repo(self, repo_root: pathlib.Path) -> None:
        (repo_root / "providers" / "opencode").mkdir(parents=True)
        (repo_root / "providers" / "opencode" / "provider.toml").write_text(
            'api = "https://opencode.ai/zen/v1"\\nenv = ["OPENCODE_API_KEY"]\\n',
            encoding="utf-8",
        )

    def test_opencode_probes_carry_random_session_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            self._prepare_opencode_repo(repo_root)
            seen_headers: list[dict] = []

            def fake_urlopen(request, timeout=None):
                seen_headers.append(dict(request.header_items()))

                class _FakeResponse:
                    status = 200

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def read(self):
                        return json.dumps({"output_text": "Muse Spark 1.2"}).encode("utf-8")

                return _FakeResponse()

            with mock.patch.object(tfm.urllib.request, "urlopen", side_effect=fake_urlopen):
                outcome = tfm.test_triple("https://opencode.ai/zen/v1", "key", "muse-spark-1.2-contributor-free")
            self.assertIsNotNone(outcome)
            self.assertEqual(len(seen_headers), 1)
            lowered = {key.lower(): value for key, value in seen_headers[0].items()}
            session = lowered.get("x-opencode-session") or ""
            self.assertIsInstance(session, str)
            # Zen free-tier gate validates the canonical session shape.
            self.assertRegex(session, r"^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$")
            self.assertEqual(lowered.get("x-opencode-client"), "cli")
            self.assertTrue(lowered.get("x-opencode-request", "").startswith("msg_"))

    def test_opencode_payload_gains_gate_stream_and_tools(self):
        seen_payloads: list[dict] = []

        def fake_urlopen(request, timeout=None):
            seen_payloads.append(json.loads(request.data.decode("utf-8")))

            class _FakeResponse:
                status = 200

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return json.dumps({"output_text": "Muse Spark 1.2"}).encode("utf-8")

            return _FakeResponse()

        with mock.patch.object(tfm.urllib.request, "urlopen", side_effect=fake_urlopen):
            tfm.test_triple("https://opencode.ai/zen/v1", "key", "muse-spark-1.2-contributor-free")
        self.assertEqual(len(seen_payloads), 1)
        payload = seen_payloads[0]
        self.assertIs(payload["stream"], True)
        self.assertEqual([t["name"] for t in payload["tools"]], ["bash", "glob", "grep", "read"])

    def test_other_providers_payload_untouched(self):
        seen_payloads: list[dict] = []

        def fake_urlopen(request, timeout=None):
            seen_payloads.append(json.loads(request.data.decode("utf-8")))

            class _FakeResponse:
                status = 200

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return json.dumps({"output_text": "ok"}).encode("utf-8")

            return _FakeResponse()

        with mock.patch.object(tfm.urllib.request, "urlopen", side_effect=fake_urlopen):
            tfm.test_triple("https://zenmux.ai/api/v1", "key", "model-a")
        self.assertEqual(len(seen_payloads), 1)
        self.assertNotIn("stream", seen_payloads[0])
        self.assertNotIn("tools", seen_payloads[0])

    def test_opencode_session_values_differ_between_probes(self):
        seen: list[str] = []

        def fake_urlopen(request, timeout=None):
            lowered = {key.lower(): value for key, value in dict(request.header_items()).items()}
            seen.append(lowered.get("x-opencode-session", ""))

            class _FakeResponse:
                status = 200

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return json.dumps({"output_text": "ok"}).encode("utf-8")

            return _FakeResponse()

        with mock.patch.object(tfm.urllib.request, "urlopen", side_effect=fake_urlopen):
            tfm.test_triple("https://opencode.ai/zen/v1", "key", "model-a")
            tfm.test_triple("https://opencode.ai/zen/v1", "key", "model-b")
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(seen))
        self.assertNotEqual(seen[0], seen[1])

    def test_other_providers_send_no_session_header(self):
        seen_headers: list[dict] = []

        def fake_urlopen(request, timeout=None):
            seen_headers.append(dict(request.header_items()))

            class _FakeResponse:
                status = 200

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return json.dumps({"output_text": "ok"}).encode("utf-8")

            return _FakeResponse()

        with mock.patch.object(tfm.urllib.request, "urlopen", side_effect=fake_urlopen):
            tfm.test_triple("https://zenmux.ai/api/v1", "key", "model-a")
        self.assertEqual(len(seen_headers), 1)
        lowered = {key.lower() for key in seen_headers[0]}
        self.assertNotIn("x-opencode-session", lowered)

    def test_opencode_probes_carry_opencode_user_agent(self):
        seen_headers: list[dict] = []

        def fake_urlopen(request, timeout=None):
            seen_headers.append(dict(request.header_items()))

            class _FakeResponse:
                status = 200

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return json.dumps({"output_text": "ok"}).encode("utf-8")

            return _FakeResponse()

        with mock.patch.object(tfm.urllib.request, "urlopen", side_effect=fake_urlopen):
            tfm.test_triple("https://opencode.ai/zen/v1", "key", "model-a")
        self.assertEqual(len(seen_headers), 1)
        lowered = {key.lower(): value for key, value in seen_headers[0].items()}
        # UA must match the official client shape (gate requires >= 1.17.0):
        # opencode/<channel>/<version>/<client>
        self.assertEqual(
            lowered.get("user-agent"), f"opencode/{tfm.OPENCODE_CHANNEL}/{tfm.OPENCODE_VERSION}/{tfm.OPENCODE_CLIENT}"
        )

    def test_other_providers_send_no_explicit_user_agent(self):
        seen_headers: list[dict] = []

        def fake_urlopen(request, timeout=None):
            seen_headers.append(dict(request.header_items()))

            class _FakeResponse:
                status = 200

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self):
                    return json.dumps({"output_text": "ok"}).encode("utf-8")

            return _FakeResponse()

        with mock.patch.object(tfm.urllib.request, "urlopen", side_effect=fake_urlopen):
            tfm.test_triple("https://zenmux.ai/api/v1", "key", "model-a")
        self.assertEqual(len(seen_headers), 1)
        lowered = {key.lower() for key in seen_headers[0]}
        self.assertNotIn("user-agent", lowered)


class ResolveApiKeysTest(unittest.TestCase):
    def test_parses_bulk_secret(self):
        env = {tfm.BULK_KEYS_ENV: json.dumps({"ZENMUX_API_KEY": " a ", "NAN_API_KEY": " n "})}
        self.assertEqual(tfm.resolve_api_keys(env), {"ZENMUX_API_KEY": "a", "NAN_API_KEY": "n"})

    def test_missing_bulk_env_is_fatal(self):
        with self.assertRaises(RuntimeError):
            tfm.resolve_api_keys({})

    def test_invalid_json_is_fatal(self):
        with self.assertRaises(RuntimeError):
            tfm.resolve_api_keys({tfm.BULK_KEYS_ENV: "{not json"})

    def test_non_object_json_is_fatal(self):
        with self.assertRaises(RuntimeError):
            tfm.resolve_api_keys({tfm.BULK_KEYS_ENV: "[]"})

    def test_blank_values_are_dropped(self):
        tfm.resolve_api_keys({tfm.BULK_KEYS_ENV: json.dumps({"A_API_KEY": "", "B_API_KEY": "  "})})
        self.assertEqual(tfm.resolve_api_keys({tfm.BULK_KEYS_ENV: "{}"}), {})
        self.assertEqual(tfm.resolve_api_keys({tfm.BULK_KEYS_ENV: json.dumps({"A_API_KEY": "x"})}), {"A_API_KEY": "x"})

    def test_no_fallback_to_individual_env_vars(self):
        with self.assertRaises(RuntimeError):
            tfm.test_free_models(
                pathlib.Path("."),
                pareto_fixture(),
                env={"ZENMUX_API_KEY": "individual-env-value"},
                tester=lambda api, key, model: (1, "responses"),
            )


class OutputLoadingTest(unittest.TestCase):
    def test_load_output_missing_file_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(tfm.load_output(pathlib.Path(tmp) / "missing.json"), {})

    def test_load_output_invalid_json_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(tfm.load_output(path), {})

    def test_load_output_non_object_yields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "array.json"
            path.write_text("[]", encoding="utf-8")
            self.assertEqual(tfm.load_output(path), {})

    def test_main_merges_into_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = pathlib.Path(tmp)
            pareto_path = repo_root / "pareto.json"
            output_path = repo_root / "tested_models.json"
            pareto_path.write_text(json.dumps(pareto_fixture()), encoding="utf-8")
            output_path.write_text(
                json.dumps({"paid": {"keep": True}}),
                encoding="utf-8",
            )
            free_result = {
                "free": {
                    "openai/gpt-5": {
                        "providers": {"inferx": {"openai/gpt-5": {"latency_ms": 1, "endpoint_type": "responses"}}}
                    }
                }
            }
            with (
                mock.patch.object(tfm, "REPO_ROOT", repo_root),
                mock.patch.object(tfm, "PARETO_PATH", pareto_path),
                mock.patch.object(tfm, "OUTPUT_PATH", output_path),
                mock.patch.object(
                    tfm, "test_free_models", return_value={"paid": {"keep": True}, **free_result}
                ) as tester,
                mock.patch.dict(os.environ, {tfm.BULK_KEYS_ENV: "{}"}),
            ):
                tfm.main()
            merged = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(merged, {"paid": {"keep": True}, **free_result, "providers": []})
            # existing payload was passed into test_free_models
            self.assertEqual(tester.call_args.kwargs["existing"], {"paid": {"keep": True}})


if __name__ == "__main__":
    unittest.main()
