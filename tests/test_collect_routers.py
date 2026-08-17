#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "collect_routers.py"
ROUTER_PROVIDERS = {
    "abacus",
    "aihubmix",
    "edenai",
    "github-copilot",
    "gitlab",
    "kilo",
    "nano-gpt",
    "openrouter",
    "pioneer",
    "requesty",
    "vercel",
}
NON_ROUTER_PROVIDERS = {
    "amazon-bedrock",
    "anthropic",
    "deepseek",
    "google",
    "hpc-ai",
    "huggingface",
    "llama",
    "meta",
    "nvidia",
    "openai",
    "tensorx",
    "wafer.ai",
    "watsonx",
    "xiaomi",
    "zai",
    "zhipuai",
}
EXPECTED_DECISIONS = {
    **dict.fromkeys(ROUTER_PROVIDERS, True),
    **dict.fromkeys(NON_ROUTER_PROVIDERS, False),
}


def load_script():
    spec = importlib.util.spec_from_file_location("collect_routers", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect_routers = load_script()


class CollectRoutersTest(unittest.TestCase):
    def test_curated_provider_fixtures_exist(self):
        provider_files = {path.parent.name for path in (REPO_ROOT / "providers").glob("*/provider.toml")}
        self.assertTrue(ROUTER_PROVIDERS <= provider_files)
        self.assertTrue(NON_ROUTER_PROVIDERS <= provider_files)

    def test_prompt_uses_generic_hosting_criteria(self):
        messages = collect_routers.build_prompt(
            "example-provider",
            'name = "Example Provider"\n',
            None,
            None,
            "Example documentation evidence.",
        )
        system_prompt = messages[0]["content"]
        self.assertIn("PURE_ROUTER", system_prompt)
        self.assertIn("MANAGED_MODEL_PLATFORM", system_prompt)
        self.assertIn("underlying inference", system_prompt)
        self.assertNotIn("Amazon Bedrock", system_prompt)

    def test_forced_managed_providers_are_always_not_routers(self):
        expected = {
            "azure",
            "azure-cognitive-services",
            "amazon-bedrock",
            "google-vertex",
            "google-vertex-anthropic",
            "snowflake-cortex",
            "huggingface",
        }
        self.assertEqual(collect_routers.FORCED_NON_ROUTER_PROVIDERS, expected)
        for provider_slug in expected:
            decision = collect_routers.forced_non_router_decision(provider_slug)
            self.assertEqual(decision["is_router"], False)
            self.assertTrue(decision["reason"].strip())
        self.assertIsNone(collect_routers.forced_non_router_decision("openrouter"))
        with mock.patch.object(
            collect_routers.urllib.request,
            "urlopen",
            side_effect=AssertionError("forced providers must not call AI"),
        ):
            decision = collect_routers.classify_provider(
                "amazon-bedrock",
                'name = "Amazon Bedrock"\n',
                None,
                None,
                "",
                "model",
                "https://example.invalid",
                "[REDACTED]",
            )
        self.assertFalse(decision["is_router"])

    def test_forced_router_providers_are_always_routers(self):
        expected = {"pioneer"}
        self.assertEqual(collect_routers.FORCED_ROUTER_PROVIDERS, expected)
        for provider_slug in expected:
            decision = collect_routers.forced_router_decision(provider_slug)
            self.assertEqual(decision["is_router"], True)
            self.assertTrue(decision["reason"].strip())
        self.assertIsNone(collect_routers.forced_router_decision("openrouter"))
        with mock.patch.object(
            collect_routers.urllib.request,
            "urlopen",
            side_effect=AssertionError("forced providers must not call AI"),
        ):
            decision = collect_routers.classify_provider(
                "pioneer",
                'name = "Pioneer"\n',
                None,
                None,
                "",
                "model",
                "https://example.invalid",
                "[REDACTED]",
            )
        self.assertTrue(decision["is_router"])

    def test_frontier_model_detection_excludes_open_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            provider_models = root / "providers" / "example" / "models"
            models_root = root / "models"
            provider_models.mkdir(parents=True)
            (models_root / "anthropic").mkdir(parents=True)
            (models_root / "openai").mkdir(parents=True)
            (models_root / "anthropic" / "claude-opus-4-8.toml").write_text("open_weights = false\n", encoding="utf-8")
            (models_root / "openai" / "gpt-oss-120b.toml").write_text("open_weights = true\n", encoding="utf-8")
            (provider_models / "claude.toml").write_text('base_model = "anthropic/claude-opus-4-8"\n', encoding="utf-8")
            (provider_models / "gpt-oss.toml").write_text('base_model = "openai/gpt-oss-120b"\n', encoding="utf-8")

            signal = collect_routers.inspect_frontier_models(provider_models, models_root)

        self.assertTrue(signal["has_closed_frontier_models"])
        self.assertEqual(signal["models"], ["anthropic/claude-opus-4-8"])

    def test_prompt_includes_closed_frontier_model_signal(self):
        messages = collect_routers.build_prompt(
            "example-provider",
            'name = "Example Provider"\n',
            None,
            None,
            "Example documentation evidence.",
            "closed-source frontier models: anthropic/claude-opus-4-8",
        )
        user_prompt = messages[1]["content"]
        self.assertIn("closed-source frontier models", user_prompt)
        self.assertIn("not sufficient", messages[0]["content"])

    def test_provider_has_base_models_requires_base_model_field(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            model_dir = root / "models"
            model_dir.mkdir()
            (model_dir / "open-model.toml").write_text('name = "Open model"\nopen_weights = true\n', encoding="utf-8")
            self.assertFalse(collect_routers.provider_has_base_models(model_dir))
            catalog_models = root / "catalog-models" / "example"
            catalog_models.mkdir(parents=True)
            (catalog_models / "open-model.toml").write_text(
                'name = "Inline model"\nopen_weights = true\n', encoding="utf-8"
            )
            self.assertTrue(collect_routers.provider_has_catalog_models(model_dir, "example", root / "catalog-models"))
            (model_dir / "relayed-model.toml").write_text(
                'base_model = "anthropic/claude-opus-4-8"\n', encoding="utf-8"
            )
            self.assertTrue(collect_routers.provider_has_base_models(model_dir))

    def test_debug_usage_reports_openrouter_cost(self):
        debug_text = collect_routers.usage_debug_text(
            "demo",
            {
                "prompt_tokens": 1000,
                "completion_tokens": 250,
                "total_tokens": 1250,
                "cost": "0.0012345",
                "cost_details": {"upstream_inference_cost": 0.0012},
            },
        )
        self.assertIn("prompt_tokens: 1000", debug_text)
        self.assertIn("completion_tokens: 250", debug_text)
        self.assertIn("cost_usd: $0.00123450 (usage.cost)", debug_text)
        self.assertIn("upstream_inference_cost_usd: $0.00120000", debug_text)

    def test_main_writes_forced_decisions_without_ai_or_documentation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            provider_root = root / "providers"
            output_dir = root / "routers"
            forced_provider_slugs = (
                collect_routers.FORCED_NON_ROUTER_PROVIDERS | collect_routers.FORCED_ROUTER_PROVIDERS
            )
            for provider_slug in forced_provider_slugs:
                provider_dir = provider_root / provider_slug
                provider_dir.mkdir(parents=True)
                (provider_dir / "provider.toml").write_text(f'name = "{provider_slug}"\n', encoding="utf-8")
                (provider_dir / "models").mkdir()
                (provider_dir / "models" / "model.toml").write_text(
                    'base_model = "anthropic/claude-opus-4-8"\n', encoding="utf-8"
                )
            argv = [
                str(SCRIPT_PATH),
                "--provider-dir",
                str(provider_root),
                "--output-dir",
                str(output_dir),
            ]
            for provider_slug in sorted(forced_provider_slugs):
                argv.extend(("--provider", provider_slug))

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(collect_routers, "load_cache", return_value={}),
                mock.patch.object(collect_routers, "save_cache"),
                mock.patch.object(
                    collect_routers,
                    "collect_document_context",
                    side_effect=AssertionError("forced providers must skip documentation"),
                ),
            ):
                self.assertEqual(collect_routers.main(), 0)

            for provider_slug in forced_provider_slugs:
                document = tomllib.loads((output_dir / f"{provider_slug}.toml").read_text(encoding="utf-8"))
                expected_decision = collect_routers.forced_provider_decision(provider_slug)
                self.assertIsNotNone(expected_decision)
                self.assertEqual(document["is_router"], expected_decision["is_router"])
                self.assertEqual(document["reason"], expected_decision["reason"])

    def test_main_rejects_provider_without_closed_frontier_models(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            provider_root = root / "providers" / "open-only"
            models_root = root / "models" / "openai"
            provider_root.mkdir(parents=True)
            models_root.mkdir(parents=True)
            (provider_root / "provider.toml").write_text('name = "Open Only"\n', encoding="utf-8")
            (provider_root / "models").mkdir()
            (provider_root / "models" / "gpt-oss.toml").write_text(
                'base_model = "openai/gpt-oss-120b"\n', encoding="utf-8"
            )
            (models_root / "gpt-oss-120b.toml").write_text("open_weights = true\n", encoding="utf-8")
            output_dir = root / "routers"
            argv = [
                str(SCRIPT_PATH),
                "--provider-dir",
                str(root / "providers"),
                "--output-dir",
                str(output_dir),
                "--provider",
                "open-only",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(collect_routers, "load_cache", return_value={}),
                mock.patch.object(collect_routers, "save_cache"),
                mock.patch.object(
                    collect_routers,
                    "collect_document_context",
                    side_effect=AssertionError("no-frontier providers must skip AI"),
                ),
            ):
                self.assertEqual(collect_routers.main(), 0)

            document = tomllib.loads((output_dir / "open-only.toml").read_text(encoding="utf-8"))
            self.assertFalse(document["is_router"])
            self.assertIn("closed-source frontier", document["reason"])

    def test_main_skips_provider_without_base_models(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            provider_dir = root / "providers" / "no-base-model"
            provider_dir.mkdir(parents=True)
            (provider_dir / "provider.toml").write_text('name = "No Base Model"\n', encoding="utf-8")
            model_dir = provider_dir / "models"
            model_dir.mkdir()
            (model_dir / "standalone.toml").write_text(
                'name = "Standalone model"\nopen_weights = true\n',
                encoding="utf-8",
            )
            output_dir = root / "routers"
            argv = [
                str(SCRIPT_PATH),
                "--provider-dir",
                str(root / "providers"),
                "--output-dir",
                str(output_dir),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(collect_routers, "load_cache", return_value={}),
                mock.patch.object(collect_routers, "save_cache"),
                mock.patch.object(
                    collect_routers,
                    "collect_document_context",
                    side_effect=AssertionError("providers without base_model must be skipped"),
                ),
            ):
                self.assertEqual(collect_routers.main(), 0)

            self.assertFalse((output_dir / "no-base-model.toml").exists())

    @unittest.skipUnless(
        os.environ.get("OPENROUTER_API_KEY"),
        "OPENROUTER_API_KEY is required for the live AI integration test",
    )
    def test_real_ai_classifies_curated_providers_and_writes_tomls(self):
        provider_slugs = sorted(EXPECTED_DECISIONS)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = pathlib.Path(temporary_directory) / "routers"
            argv = [
                str(SCRIPT_PATH),
                "--provider-dir",
                str(REPO_ROOT / "providers"),
                "--output-dir",
                str(output_dir),
                "--clear-cache",
            ]
            for provider_slug in provider_slugs:
                argv.extend(("--provider", provider_slug))

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(collect_routers, "save_cache"),
            ):
                self.assertEqual(collect_routers.main(), 0)

            for provider_slug, expected_is_router in EXPECTED_DECISIONS.items():
                output_path = output_dir / f"{provider_slug}.toml"
                self.assertTrue(output_path.is_file(), output_path)
                document = tomllib.loads(output_path.read_text(encoding="utf-8"))
                self.assertEqual(document["is_router"], expected_is_router)
                self.assertTrue(document["reason"].strip())


if __name__ == "__main__":
    unittest.main()
