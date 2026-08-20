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
