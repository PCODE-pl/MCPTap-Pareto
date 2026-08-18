#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / ".github" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.provider_model_stats import build_provider_outputs, collect_model_endpoints  # noqa: E402


class ProviderModelStatsLibraryTest(unittest.TestCase):
    def test_collects_model_endpoints_through_shared_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            models_dir = pathlib.Path(temporary_directory)
            model = models_dir / "acme" / "model.toml"
            model.parent.mkdir(parents=True)
            model.write_text('base_model = "acme/model"\n', encoding="utf-8")
            calls: list[tuple[str, str]] = []

            def fetch(api_url: str, model_id: str):
                calls.append((api_url, model_id))
                return [{"provider_name": "acme"}]

            models, parse_errors, endpoint_cache, api_errors = collect_model_endpoints(
                models_dir=models_dir,
                api_url="https://example.test/models",
                fetch_model_endpoints=fetch,
            )

        self.assertEqual(parse_errors, [])
        self.assertEqual(api_errors, [])
        self.assertEqual(calls, [("https://example.test/models", "acme/model")])
        self.assertEqual(endpoint_cache, {"acme/model": [{"provider_name": "acme"}]})
        self.assertEqual(models["acme/model"][0]["model_id"], "acme/model")

    def test_builds_outputs_with_provider_specific_resolver_and_stats_extractor(self):
        models = {
            "acme/model": [{"path": "model.toml", "model_id": "acme/model"}],
        }
        endpoint_cache = {
            "acme/model": [
                {"provider_name": "gateway", "metric": 1},
                {"provider_name": "gateway", "metric": 2},
            ]
        }

        outputs, collisions, unmatched, multiple = build_provider_outputs(
            models,
            endpoint_cache,
            resolve_provider=lambda name: "canonical" if name == "gateway" else None,
            extract_stats=lambda endpoint: {"metric": endpoint["metric"]},
            select_endpoint=lambda endpoints: endpoints[-1],
        )

        self.assertEqual(outputs, {"canonical/models/acme/model.json": {"metric": 2}})
        self.assertEqual(collisions, [])
        self.assertEqual(unmatched, [])
        self.assertEqual(multiple, ["gateway/acme/model: 2 endpoints; selected deterministic first"])


if __name__ == "__main__":
    unittest.main()
