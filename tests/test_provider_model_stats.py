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

from lib.provider_model_stats import (  # noqa: E402 # type: ignore
    build_provider_outputs,
    collect_model_endpoints,
    load_mapping_cache,
    save_mapping_cache,
    write_collected_outputs,
)


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

    def test_mapping_cache_round_trip_is_scoped_by_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            save_mapping_cache(root, {"Vercel": "vercel"}, ".vercel_provider_mapping_cache.json")

            self.assertEqual(
                load_mapping_cache(root, ".vercel_provider_mapping_cache.json"),
                {"Vercel": "vercel"},
            )
            self.assertEqual(load_mapping_cache(root, ".openrouter_provider_mapping_cache.json"), {})

    def test_clears_direct_stats_output_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            output_dir = root / "stats" / "vercel"
            stale_file = output_dir / "stale.json"
            stale_file.parent.mkdir(parents=True)
            stale_file.write_text("stale\n", encoding="utf-8")

            written, synthetic_written, errors, collisions = write_collected_outputs(
                outputs={},
                output_dir=output_dir,
                providers_dir=root / "providers",
                routers_dir=root / "routers",
                synthetic_dir=root / "stats" / "_synthetic" / "vercel",
                excluded_provider="vercel",
                dry_run=False,
            )
            stale_exists_after = stale_file.exists()

        self.assertEqual(written, 0)
        self.assertEqual(synthetic_written, 0)
        self.assertEqual(errors, [])
        self.assertEqual(collisions, [])
        self.assertFalse(stale_exists_after)

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
