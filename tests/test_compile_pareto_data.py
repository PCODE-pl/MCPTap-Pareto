#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "compile_pareto_data.py"


def load_script():
    spec = importlib.util.spec_from_file_location("compile_pareto_data", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compile_pareto_data = load_script()


class CompileParetoDataTest(unittest.TestCase):
    def test_builds_openrouter_refs_with_canonical_slugs_from_models_catalog(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            model_path = root / "providers" / "openrouter" / "models" / "google" / "model.toml"
            model_path.parent.mkdir(parents=True)
            model_path.write_text('name = "Google model"\n', encoding="utf-8")
            (root / "models.json").write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "id": "google/model",
                                "canonical_slug": "google/model-03-25",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            refs = compile_pareto_data.build_openrouter_refs(root)

        self.assertEqual(refs[0]["canonical_slug"], "google/model-03-25")

    def test_extracts_dated_slug_variants(self):
        self.assertEqual(compile_pareto_data.extract_display_date("Preview (Mar' 25)"), (3, 25))
        self.assertEqual(compile_pareto_data.extract_slug_date("google/model-05-06"), (5, 6))
        self.assertEqual(compile_pareto_data.extract_slug_date("google/model-20250325"), (3, 25))
        self.assertEqual(compile_pareto_data.extract_slug_date("google/model-2025-03-25"), (3, 25))
        self.assertIsNone(compile_pareto_data.extract_slug_date("google/model"))

    def test_resolves_dated_display_name_to_dated_slug_when_short_slug_collides(self):
        refs = [
            {
                "model_id": "google/gemini-2.5-pro-preview",
                "base_model": None,
                "family": "gemini",
                "name": "Gemini 2.5 Pro Preview 06-05",
                "canonical_slug": "google/gemini-2.5-pro-preview-06-05",
                "reasoning": True,
            },
            {
                "model_id": "google/gemini-2.5-pro-preview-05-06",
                "base_model": None,
                "family": "gemini-pro",
                "name": "Gemini 2.5 Pro Preview 05-06",
                "canonical_slug": "google/gemini-2.5-pro-preview-03-25",
                "reasoning": True,
            },
        ]
        record = {
            "model_permaslug": "google/gemini-2.5-pro-preview-unknown",
            "display_name": "Gemini 2.5 Pro Preview (Mar' 25)",
        }

        resolved = compile_pareto_data.resolve_canonical_model(record, refs, {})

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved[0], "google/gemini-2.5-pro-preview-05-06")
        self.assertEqual(resolved[1]["model_id"], "google/gemini-2.5-pro-preview-05-06")

    def test_resolves_benchmark_alias_using_display_date_when_ref_is_versioned(self):
        refs = [
            {
                "model_id": "google/gemini-2.5-pro-preview-05-06",
                "base_model": None,
                "canonical_slug": "google/gemini-2.5-pro-preview-03-25",
                "family": None,
                "name": "Gemini 2.5 Pro Preview 05-06",
                "reasoning": True,
            },
            {
                "model_id": "google/gemini-2.5-pro-preview",
                "base_model": None,
                "canonical_slug": "google/gemini-2.5-pro-preview-06-05",
                "family": None,
                "name": "Gemini 2.5 Pro Preview 06-05",
                "reasoning": True,
            },
        ]
        record = {
            "model_permaslug": "google/gemini-2.5-pro-preview-unknown",
            "display_name": "Gemini 2.5 Pro Preview (Mar' 25)",
        }

        resolved = compile_pareto_data.resolve_canonical_model(record, refs, {})

        self.assertEqual(resolved[0], "google/gemini-2.5-pro-preview-05-06")
        self.assertEqual(resolved[1]["model_id"], "google/gemini-2.5-pro-preview-05-06")

    def test_prefers_exact_model_id_before_normalized_aliases(self):
        refs = [
            {
                "model_id": "google/gemini-2.5-pro-preview",
                "base_model": None,
                "canonical_slug": "google/gemini-2.5-pro-preview-06-05",
                "family": None,
                "name": "Gemini 2.5 Pro Preview 06-05",
                "reasoning": True,
            },
            {
                "model_id": "google/gemini-2.5-pro-preview-05-06",
                "base_model": None,
                "canonical_slug": "google/gemini-2.5-pro-preview-03-25",
                "family": None,
                "name": "Gemini 2.5 Pro Preview 05-06",
                "reasoning": True,
            },
        ]

        resolved = compile_pareto_data.resolve_canonical_model(
            {"model_permaslug": "google/gemini-2.5-pro-preview"},
            refs,
            {},
        )

        self.assertEqual(resolved[0], "google/gemini-2.5-pro-preview")

    def test_attaches_average_stats_for_matching_provider_model_or_empty_stats(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            for provider in ("acme", "missing-stats"):
                model = root / "providers" / provider / "models" / "lab" / "model.toml"
                model.parent.mkdir(parents=True)
                model.write_text(
                    'base_model = "lab/model"\n[cost]\ninput = 1\noutput = 2\n',
                    encoding="utf-8",
                )

            average_stats = root / "stats" / "_average" / "acme" / "models" / "lab" / "model.json"
            average_stats.parent.mkdir(parents=True)
            average_stats.write_text(
                json.dumps({"uptime_last_1d": 99, "latency_last_1d": 120}),
                encoding="utf-8",
            )

            providers = compile_pareto_data.collect_provider_prices(
                root,
                "lab/model",
                {},
                {"family": None, "name": None, "reasoning": False},
                True,
            )

        self.assertEqual(
            providers,
            {
                "acme": {
                    "lab/model": {
                        "is_thinking": False,
                        "cost": {"input": 1, "output": 2},
                        "stats": {"uptime_last_1d": 99, "latency_last_1d": 120},
                    }
                },
                "missing-stats": {
                    "lab/model": {
                        "is_thinking": False,
                        "cost": {"input": 1, "output": 2},
                        "stats": {},
                    }
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
