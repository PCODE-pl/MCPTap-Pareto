#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import unittest

WORKFLOW_PATH = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "compile-pareto-data.yml"
WORKFLOW = WORKFLOW_PATH.read_text(encoding="utf-8")


class CompileWorkflowTest(unittest.TestCase):
    def test_commit_step_publishes_stats_with_explicit_github_credentials(self):
        self.assertIn("GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}", WORKFLOW)
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


if __name__ == "__main__":
    unittest.main()
