#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import unittest

WORKFLOW_PATH = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "compile-pareto-data.yml"
RUN_SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "run.sh"
WORKFLOW = WORKFLOW_PATH.read_text(encoding="utf-8")
RUN_SCRIPT = RUN_SCRIPT_PATH.read_text(encoding="utf-8")


class CompileWorkflowTest(unittest.TestCase):
    def test_commit_step_publishes_stats_with_explicit_github_credentials(self):
        self.assertIn("GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}", WORKFLOW)
        self.assertIn("MCPTAP_EXTRAS_TOKEN: ${{ secrets.MCPTAP_EXTRAS_TOKEN }}", WORKFLOW)
        self.assertIn(
            "token: ${{ secrets.MCPTAP_EXTRAS_TOKEN }}\n          path: mcp-tap-extras",
            WORKFLOW,
        )
        self.assertIn("A read-capable MCPTAP_EXTRAS_TOKEN secret is required.", WORKFLOW)
        self.assertIn(
            'github_auth="$(printf \'x-access-token:%s\' "$GITHUB_TOKEN" | base64 -w0)"',
            WORKFLOW,
        )
        self.assertIn(
            'git config --local http.https://github.com/.extraheader "AUTHORIZATION: basic ${github_auth}"',
            WORKFLOW,
        )
        self.assertIn(
            "git status --porcelain -- models providers stats pareto.json .requesty_ai_mapping_cache.json",
            WORKFLOW,
        )
        self.assertIn(
            "git add -A -- models providers stats pareto.json .requesty_ai_mapping_cache.json",
            WORKFLOW,
        )

    def test_provider_stats_collectors_run_in_parallel_and_all_failures_propagate(self):
        match = re.search(
            r"      - name: Get provider model stats\n(.*?)\n      - name: Compute average provider model stats",
            WORKFLOW,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        assert match is not None
        block = match.group(1)

        self.assertIn("pids=()", block)
        self.assertIn('pids+=("$!")', block)
        self.assertIn('if ! wait "$pid"; then', block)
        self.assertIn('exit "$status"', block)
        for script in (
            "get_provider_model_stats_from_openrouter.py",
            "get_provider_model_stats_from_vercel.py",
            "get_provider_model_stats_from_llmgateway.py",
        ):
            self.assertIn(f"python .github/scripts/{script} &", block)

    def test_local_runner_forwards_shared_hook_token(self):
        self.assertIn('MCPTAP_EXTRAS_TOKEN="${MCPTAP_EXTRAS_TOKEN:-$GITHUB_TOKEN}"', RUN_SCRIPT)
        self.assertIn('--secret MCPTAP_EXTRAS_TOKEN="$MCPTAP_EXTRAS_TOKEN"', RUN_SCRIPT)


if __name__ == "__main__":
    unittest.main()
