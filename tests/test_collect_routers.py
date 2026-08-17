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
    "aihubmix",
    "edenai",
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
            "Example documentation evidence.",
        )
        system_prompt = messages[0]["content"]
        self.assertIn("PURE_ROUTER", system_prompt)
        self.assertIn("MANAGED_MODEL_PLATFORM", system_prompt)
        self.assertIn("underlying model inference", system_prompt)
        self.assertIn("Do not use a provider name, slug, or memorized label", system_prompt)
        self.assertNotIn("Amazon Bedrock", system_prompt)

    def test_ai_decisions_create_toml_for_router_and_non_router_providers(self):
        provider_slugs = sorted(EXPECTED_DECISIONS)

        def fake_context(doc_url):
            return f"Evidence for {doc_url}", [doc_url] if doc_url else []

        def fake_classify(provider_slug, *_args, **_kwargs):
            return {
                "is_router": EXPECTED_DECISIONS[provider_slug],
                "reason": f"Test evidence for {provider_slug}.",
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = pathlib.Path(temporary_directory) / "routers"
            argv = [
                str(SCRIPT_PATH),
                "--provider-dir",
                str(REPO_ROOT / "providers"),
                "--output-dir",
                str(output_dir),
            ]
            for provider_slug in provider_slugs:
                argv.extend(("--provider", provider_slug))

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
                mock.patch.object(collect_routers, "collect_document_context", fake_context),
                mock.patch.object(collect_routers, "classify_provider", fake_classify),
                mock.patch.object(collect_routers, "load_cache", return_value={}),
                mock.patch.object(collect_routers, "save_cache"),
            ):
                self.assertEqual(collect_routers.main(), 0)

            for provider_slug, expected_is_router in EXPECTED_DECISIONS.items():
                output_path = output_dir / f"{provider_slug}.toml"
                self.assertTrue(output_path.is_file(), output_path)
                document = tomllib.loads(output_path.read_text(encoding="utf-8"))
                self.assertEqual(document["is_router"], expected_is_router)
                self.assertEqual(document["reason"], f"Test evidence for {provider_slug}.")


if __name__ == "__main__":
    unittest.main()
