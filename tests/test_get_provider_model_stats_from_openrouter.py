#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "get_provider_model_stats_from_openrouter.py"


def load_script():
    spec = importlib.util.spec_from_file_location("get_provider_model_stats_from_openrouter", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_provider_model_stats = load_script()


class SyntheticOpenRouterStatsTest(unittest.TestCase):
    def test_loads_only_non_openrouter_routers_enabled_for_models_directory_structure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            routers_dir = pathlib.Path(temporary_directory)
            (routers_dir / "fast-router.toml").write_text(
                "router_providers_from_models_dir_struct = true\n",
                encoding="utf-8",
            )
            (routers_dir / "openrouter.toml").write_text(
                "router_providers_from_models_dir_struct = true\n",
                encoding="utf-8",
            )
            (routers_dir / "disabled-router.toml").write_text(
                "router_providers_from_models_dir_struct = false\n",
                encoding="utf-8",
            )
            (routers_dir / "invalid.toml").write_text("[invalid\n", encoding="utf-8")

            routers, errors = get_provider_model_stats.load_router_providers_from_models_dir_struct(routers_dir)

        self.assertEqual(routers, {"fast-router"})
        self.assertEqual(len(errors), 1)
        self.assertIn("invalid.toml", errors[0])

    def test_copies_stats_for_router_model_with_matching_stats_provider_path_component(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            providers_dir = root / "providers"
            routers_dir = root / "routers"
            stats_dir = root / "stats" / "openrouter"
            synthetic_dir = root / "stats" / "_synthetic" / "openrouter"
            routers_dir.mkdir()
            (routers_dir / "fast-router.toml").write_text(
                "router_providers_from_models_dir_struct = true\n",
                encoding="utf-8",
            )
            source_stats = stats_dir / "acme" / "models" / "acme" / "base-model.json"
            source_stats.parent.mkdir(parents=True)
            source_payload = {"uptime_last_30m": 0.99, "latency_last_30m": 123}
            source_stats.write_text(json.dumps(source_payload) + "\n", encoding="utf-8")
            model_dir = providers_dir / "fast-router" / "models" / "acme" / "nested"
            model_dir.mkdir(parents=True)
            (model_dir / "proxy-model.toml").write_text('base_model = "acme/base-model"\n', encoding="utf-8")
            (model_dir / "missing-model.toml").write_text('base_model = "acme/not-found"\n', encoding="utf-8")

            written, errors, collisions = get_provider_model_stats.write_synthetic_stats(
                providers_dir=providers_dir,
                routers_dir=routers_dir,
                stats_dir=stats_dir,
                synthetic_dir=synthetic_dir,
                dry_run=False,
            )

            self.assertEqual(written, 1)
            self.assertEqual(errors, [])
            self.assertEqual(collisions, [])
            self.assertEqual(
                (synthetic_dir / "fast-router" / "models" / "acme" / "nested" / "proxy-model.json").read_text(
                    encoding="utf-8"
                ),
                source_stats.read_text(encoding="utf-8"),
            )
            self.assertFalse(
                (synthetic_dir / "fast-router" / "models" / "acme" / "nested" / "missing-model.json").exists()
            )

    def test_dry_run_reports_writes_without_creating_synthetic_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            providers_dir = root / "providers"
            routers_dir = root / "routers"
            stats_dir = root / "stats" / "openrouter"
            synthetic_dir = root / "synthetic"
            routers_dir.mkdir()
            (routers_dir / "fast-router.toml").write_text(
                "router_providers_from_models_dir_struct = true\n",
                encoding="utf-8",
            )
            source_stats = stats_dir / "acme" / "models" / "acme" / "model.json"
            source_stats.parent.mkdir(parents=True)
            source_stats.write_text('{"uptime_last_30m": 1}\n', encoding="utf-8")
            model = providers_dir / "fast-router" / "models" / "acme" / "model.toml"
            model.parent.mkdir(parents=True)
            model.write_text('base_model = "acme/model"\n', encoding="utf-8")

            written, errors, collisions = get_provider_model_stats.write_synthetic_stats(
                providers_dir=providers_dir,
                routers_dir=routers_dir,
                stats_dir=stats_dir,
                synthetic_dir=synthetic_dir,
                dry_run=True,
            )

            self.assertEqual(written, 1)
            self.assertEqual(errors, [])
            self.assertEqual(collisions, [])
            self.assertFalse(synthetic_dir.exists())

    def test_skips_collision_and_reports_all_sources(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            providers_dir = root / "providers"
            routers_dir = root / "routers"
            stats_dir = root / "stats" / "openrouter"
            synthetic_dir = root / "synthetic"
            routers_dir.mkdir()
            (routers_dir / "fast-router.toml").write_text(
                "router_providers_from_models_dir_struct = true\n",
                encoding="utf-8",
            )
            for stats_provider, payload in (("deepinfra", "first"), ("moonshotai", "second")):
                source = stats_dir / stats_provider / "models" / "acme" / "base.json"
                source.parent.mkdir(parents=True)
                source.write_text(f'{{"source": "{payload}"}}\n', encoding="utf-8")
            model = providers_dir / "fast-router" / "models" / "deepinfra" / "moonshotai" / "model.toml"
            model.parent.mkdir(parents=True)
            model.write_text('base_model = "acme/base"\n', encoding="utf-8")

            written, errors, collisions = get_provider_model_stats.write_synthetic_stats(
                providers_dir=providers_dir,
                routers_dir=routers_dir,
                stats_dir=stats_dir,
                synthetic_dir=synthetic_dir,
                dry_run=False,
            )

            self.assertEqual(written, 0)
            self.assertEqual(errors, [])
            self.assertEqual(len(collisions), 1)
            self.assertIn("fast-router/models/deepinfra/moonshotai/model.json", collisions[0])
            self.assertIn("deepinfra/models/acme/base.json", collisions[0])
            self.assertIn("moonshotai/models/acme/base.json", collisions[0])
            self.assertFalse(synthetic_dir.exists())

    def test_rejects_base_models_that_escape_the_stats_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            providers_dir = root / "providers"
            routers_dir = root / "routers"
            stats_dir = root / "stats" / "openrouter"
            synthetic_dir = root / "synthetic"
            routers_dir.mkdir()
            (routers_dir / "fast-router.toml").write_text(
                "router_providers_from_models_dir_struct = true\n",
                encoding="utf-8",
            )
            (stats_dir / "acme" / "models").mkdir(parents=True)
            secret = root / "secret.json"
            secret.write_text('{"must_not_copy": true}\n', encoding="utf-8")
            model_dir = providers_dir / "fast-router" / "models" / "acme"
            model_dir.mkdir(parents=True)
            (model_dir / "traversal.toml").write_text(
                'base_model = "../../../../secret"\n',
                encoding="utf-8",
            )
            (model_dir / "absolute.toml").write_text(
                f'base_model = "acme/{secret}"\n',
                encoding="utf-8",
            )

            written, errors, collisions = get_provider_model_stats.write_synthetic_stats(
                providers_dir=providers_dir,
                routers_dir=routers_dir,
                stats_dir=stats_dir,
                synthetic_dir=synthetic_dir,
                dry_run=False,
            )

            self.assertEqual(written, 0)
            self.assertEqual(errors, [])
            self.assertEqual(collisions, [])
            self.assertFalse(synthetic_dir.exists())

    def test_rejects_symlinked_stats_outside_router_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            providers_dir = root / "providers"
            routers_dir = root / "routers"
            stats_dir = root / "stats" / "openrouter"
            synthetic_dir = root / "synthetic"
            routers_dir.mkdir()
            (routers_dir / "fast-router.toml").write_text(
                "router_providers_from_models_dir_struct = true\n",
                encoding="utf-8",
            )
            source_dir = stats_dir / "acme" / "models" / "acme"
            source_dir.mkdir(parents=True)
            outside_stats = root / "outside-stats.json"
            outside_stats.write_text('{"must_not_copy": true}\n', encoding="utf-8")
            (source_dir / "model.json").symlink_to(outside_stats)
            model = providers_dir / "fast-router" / "models" / "acme" / "model.toml"
            model.parent.mkdir(parents=True)
            model.write_text('base_model = "acme/model"\n', encoding="utf-8")

            written, errors, collisions = get_provider_model_stats.write_synthetic_stats(
                providers_dir=providers_dir,
                routers_dir=routers_dir,
                stats_dir=stats_dir,
                synthetic_dir=synthetic_dir,
                dry_run=False,
            )

            self.assertEqual(written, 0)
            self.assertEqual(errors, [])
            self.assertEqual(collisions, [])
            self.assertFalse(synthetic_dir.exists())

    def test_rejects_windows_style_path_separators_in_base_model(self):
        self.assertIsNone(get_provider_model_stats.parse_base_model(r"acme/..\..\secret"))


if __name__ == "__main__":
    unittest.main()
