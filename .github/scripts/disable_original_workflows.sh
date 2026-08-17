#!/bin/bash

set -euo pipefail

gh workflow disable validate.yml --repo PCODE-pl/MCPTap-Pareto
gh workflow disable sync-models.yml --repo PCODE-pl/MCPTap-Pareto
gh workflow disable deploy.yml --repo PCODE-pl/MCPTap-Pareto
gh workflow disable publish-sdk.yml --repo PCODE-pl/MCPTap-Pareto
gh workflow disable opencode.yml --repo PCODE-pl/MCPTap-Pareto
gh workflow disable close-stale-pull-requests.yml --repo PCODE-pl/MCPTap-Pareto
gh workflow disable issue-fixer.yml --repo PCODE-pl/MCPTap-Pareto
gh workflow disable pr-reviewer.yml --repo PCODE-pl/MCPTap-Pareto
gh workflow disable ci-fixer.yml --repo PCODE-pl/MCPTap-Pareto
