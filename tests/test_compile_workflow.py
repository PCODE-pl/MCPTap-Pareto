#!/usr/bin/env python3
from __future__ import annotations

import pathlib
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

    def test_local_runner_forwards_shared_hook_token(self):
        self.assertIn('MCPTAP_EXTRAS_TOKEN="${MCPTAP_EXTRAS_TOKEN:-$GITHUB_TOKEN}"', RUN_SCRIPT)
        self.assertIn('--secret MCPTAP_EXTRAS_TOKEN="$MCPTAP_EXTRAS_TOKEN"', RUN_SCRIPT)


if __name__ == "__main__":
    unittest.main()
