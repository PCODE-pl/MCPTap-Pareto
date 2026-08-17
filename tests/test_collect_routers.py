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
    "kilo",
    "nano-gpt",
    "openrouter",
    "requesty",
}
NON_ROUTER_PROVIDERS = {
    "amazon-bedrock",
    "anthropic",
    "deepseek",
    "google",
    "openai",
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
        }
        self.assertEqual(collect_routers.FORCED_NON_ROUTER_PROVIDERS, expected)
        for provider_slug in expected:
            decision = collect_routers.forced_non_router_decision(provider_slug)
            self.assertEqual(decision["is_router"], False)
            self.assertTrue(decision["reason"].strip())
        self.assertIsNone(collect_routers.forced_non_router_decision("openrouter"))

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
            for provider_slug in collect_routers.FORCED_NON_ROUTER_PROVIDERS:
                provider_dir = provider_root / provider_slug
                provider_dir.mkdir(parents=True)
                (provider_dir / "provider.toml").write_text(f'name = "{provider_slug}"\n', encoding="utf-8")
            argv = [
                str(SCRIPT_PATH),
                "--provider-dir",
                str(provider_root),
                "--output-dir",
                str(output_dir),
            ]
            for provider_slug in sorted(collect_routers.FORCED_NON_ROUTER_PROVIDERS):
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

            for provider_slug in collect_routers.FORCED_NON_ROUTER_PROVIDERS:
                document = tomllib.loads((output_dir / f"{provider_slug}.toml").read_text(encoding="utf-8"))
                self.assertFalse(document["is_router"])
                self.assertIn("managed model-hosting platform", document["reason"])

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
