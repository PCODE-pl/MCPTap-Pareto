#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "sync_requesty_models.py"


def load_script():
    spec = importlib.util.spec_from_file_location("sync_requesty_models", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync_requesty_models = load_script()


class SyncRequestyModelsTest(unittest.TestCase):
    def test_display_name_falls_back_to_record_description(self):
        self.assertEqual(
            sync_requesty_models.display_name_for_record(
                {"description": "Requesty model description"},
                "openai",
                "gpt-5.4",
            ),
            "Requesty model description",
        )

    def test_display_name_prefers_name_and_then_display_name(self):
        self.assertEqual(
            sync_requesty_models.display_name_for_record(
                {"name": "Name", "display_name": "Display", "description": "Description"},
                "openai",
                "gpt-5.4",
            ),
            "Name",
        )
        self.assertEqual(
            sync_requesty_models.display_name_for_record(
                {"display_name": "Display", "description": "Description"},
                "openai",
                "gpt-5.4",
            ),
            "Display",
        )

    def test_display_name_falls_back_to_lab_and_model_name(self):
        self.assertEqual(
            sync_requesty_models.display_name_for_record({}, "openai", "gpt-5.4"),
            "openai/gpt-5.4",
        )

    def test_model_path_requires_a_nested_provider_directory(self):
        self.assertIsNone(sync_requesty_models.model_path("gpt-5.4"))
        self.assertEqual(
            sync_requesty_models.model_path("deepinfra/gpt-5.4").as_posix(),
            "deepinfra/gpt-5.4.toml",
        )

    def test_direct_output_dir_argument_is_not_supported(self):
        with (
            mock.patch.object(sys, "argv", [str(SCRIPT_PATH), "--output-dir", "/tmp/other"]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            sync_requesty_models.parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_sync_preserves_direct_files_and_prunes_only_nested_generated_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            canonical = root / "models" / "openai" / "gpt-5.4.toml"
            canonical.parent.mkdir(parents=True)
            canonical.write_text('name = "GPT-5.4"\n', encoding="utf-8")

            output_dir = root / "providers" / "requesty" / "models"
            output_dir.mkdir(parents=True)
            direct_file = output_dir / "gpt-5.4.toml"
            direct_content = f"{sync_requesty_models.AUTO_SOURCE_MARKER}\nmanual = true\n"
            direct_file.write_text(direct_content, encoding="utf-8")
            stale_file = output_dir / "legacy" / "stale.toml"
            stale_file.parent.mkdir()
            stale_file.write_text(f"{sync_requesty_models.AUTO_SOURCE_MARKER}\n", encoding="utf-8")

            summary = sync_requesty_models.sync(
                root,
                [
                    {
                        "id": "gpt-5.4",
                        "model_lab": "openai",
                        "model_canonical_name": "gpt-5.4",
                        "input_price": 0.000001,
                        "output_price": 0.000002,
                    },
                    {
                        "id": "deepinfra/gpt-5.4",
                        "model_lab": "openai",
                        "model_canonical_name": "gpt-5.4",
                        "input_price": 0.000001,
                        "output_price": 0.000002,
                    },
                ],
                False,
                disable_ai=True,
            )

            self.assertIn("gpt-5.4", summary["skipped_invalid_id"])
            self.assertEqual(summary["written"], ["deepinfra/gpt-5.4.toml"])
            self.assertEqual(direct_file.read_text(encoding="utf-8"), direct_content)
            self.assertTrue((output_dir / "deepinfra" / "gpt-5.4.toml").is_file())
            self.assertFalse(stale_file.exists())

    def test_sync_rejects_symlinked_output_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            canonical = root / "models" / "openai" / "gpt-5.4.toml"
            canonical.parent.mkdir(parents=True)
            canonical.write_text('name = "GPT-5.4"\n', encoding="utf-8")

            output_dir = root / "providers" / "requesty" / "models"
            output_dir.mkdir(parents=True)
            outside = root / "outside"
            outside.mkdir()
            (output_dir / "escape").symlink_to(outside, target_is_directory=True)

            summary = sync_requesty_models.sync(
                root,
                [
                    {
                        "id": "escape/gpt-5.4",
                        "model_lab": "openai",
                        "model_canonical_name": "gpt-5.4",
                        "input_price": 0.000001,
                        "output_price": 0.000002,
                    }
                ],
                False,
                disable_ai=True,
            )

            self.assertEqual(summary["written"], [])
            self.assertEqual(summary["skipped_invalid_id"], ["escape/gpt-5.4"])
            self.assertFalse((outside / "gpt-5.4.toml").exists())


if __name__ == "__main__":
    unittest.main()
