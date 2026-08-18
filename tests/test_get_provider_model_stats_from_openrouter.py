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
    def test_loads_only_routers_enabled_for_models_directory_structure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            routers_dir = pathlib.Path(temporary_directory)
            (routers_dir / "fast-router.toml").write_text(
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

    def test_copies_matching_stats_for_router_provider_and_matching_nested_path(self):
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
            (routers_dir / "openrouter.toml").write_text(
                "router_providers_from_models_dir_struct = true\n",
                encoding="utf-8",
            )

            source_stats = stats_dir / "fastrouter" / "models" / "acme" / "base-model.json"
            source_stats.parent.mkdir(parents=True)
            source_payload = {"uptime_last_30m": 0.99, "latency_last_30m": 123}
            source_stats.write_text(json.dumps(source_payload) + "\n", encoding="utf-8")

            nested_model = providers_dir / "alpha" / "models" / "fast-router" / "nested" / "proxy-model.toml"
            nested_model.parent.mkdir(parents=True)
            nested_model.write_text('base_model = "acme/base-model"\n', encoding="utf-8")

            router_model = providers_dir / "fast-router" / "models" / "arbitrary" / "router-model.toml"
            router_model.parent.mkdir(parents=True)
            router_model.write_text('base_model = "acme/base-model"\n', encoding="utf-8")

            ignored_model = providers_dir / "openrouter" / "models" / "fast-router" / "ignored.toml"
            ignored_model.parent.mkdir(parents=True)
            ignored_model.write_text('base_model = "acme/base-model"\n', encoding="utf-8")

            missing_model = providers_dir / "alpha" / "models" / "fast-router" / "missing.toml"
            missing_model.write_text('base_model = "acme/not-found"\n', encoding="utf-8")

            written, errors = get_provider_model_stats.write_synthetic_stats(
                providers_dir=providers_dir,
                routers_dir=routers_dir,
                stats_dir=stats_dir,
                synthetic_dir=synthetic_dir,
                dry_run=False,
            )

            self.assertEqual(written, 2)
            self.assertEqual(errors, [])
            self.assertEqual(
                (synthetic_dir / "alpha" / "models" / "fast-router" / "nested" / "proxy-model.json").read_text(
                    encoding="utf-8"
                ),
                source_stats.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (synthetic_dir / "fast-router" / "models" / "arbitrary" / "router-model.json").read_text(
                    encoding="utf-8"
                ),
                source_stats.read_text(encoding="utf-8"),
            )
            self.assertFalse((synthetic_dir / "openrouter").exists())
            self.assertFalse((synthetic_dir / "alpha" / "models" / "fast-router" / "missing.json").exists())

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
            source_stats = stats_dir / "fast-router" / "models" / "acme" / "model.json"
            source_stats.parent.mkdir(parents=True)
            source_stats.write_text('{"uptime_last_30m": 1}\n', encoding="utf-8")
            model = providers_dir / "alpha" / "models" / "fast-router" / "model.toml"
            model.parent.mkdir(parents=True)
            model.write_text('base_model = "acme/model"\n', encoding="utf-8")

            written, errors = get_provider_model_stats.write_synthetic_stats(
                providers_dir=providers_dir,
                routers_dir=routers_dir,
                stats_dir=stats_dir,
                synthetic_dir=synthetic_dir,
                dry_run=True,
            )

            self.assertEqual(written, 1)
            self.assertEqual(errors, [])
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
            (stats_dir / "fast-router" / "models").mkdir(parents=True)
            secret = root / "secret.json"
            secret.write_text('{"must_not_copy": true}\n', encoding="utf-8")

            model_dir = providers_dir / "alpha" / "models" / "fast-router"
            model_dir.mkdir(parents=True)
            (model_dir / "traversal.toml").write_text(
                'base_model = "../../../../secret"\n',
                encoding="utf-8",
            )
            (model_dir / "absolute.toml").write_text(
                f'base_model = "acme/{secret}"\n',
                encoding="utf-8",
            )

            written, errors = get_provider_model_stats.write_synthetic_stats(
                providers_dir=providers_dir,
                routers_dir=routers_dir,
                stats_dir=stats_dir,
                synthetic_dir=synthetic_dir,
                dry_run=False,
            )

            self.assertEqual(written, 0)
            self.assertEqual(errors, [])
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
            source_dir = stats_dir / "fast-router" / "models" / "acme"
            source_dir.mkdir(parents=True)
            outside_stats = root / "outside-stats.json"
            outside_stats.write_text('{"must_not_copy": true}\n', encoding="utf-8")
            (source_dir / "model.json").symlink_to(outside_stats)
            model = providers_dir / "alpha" / "models" / "fast-router" / "model.toml"
            model.parent.mkdir(parents=True)
            model.write_text('base_model = "acme/model"\n', encoding="utf-8")

            written, errors = get_provider_model_stats.write_synthetic_stats(
                providers_dir=providers_dir,
                routers_dir=routers_dir,
                stats_dir=stats_dir,
                synthetic_dir=synthetic_dir,
                dry_run=False,
            )

            self.assertEqual(written, 0)
            self.assertEqual(errors, [])
            self.assertFalse(synthetic_dir.exists())

    def test_rejects_windows_style_path_separators_in_base_model(self):
        self.assertIsNone(get_provider_model_stats.parse_base_model(r"acme/..\..\secret"))


if __name__ == "__main__":
    unittest.main()
