#!/bin/bash

set -euo pipefail

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  printf 'GITHUB_TOKEN must be set to allow act to push generated data.\n' >&2
  exit 1
fi

MCPTAP_EXTRAS_TOKEN="${MCPTAP_EXTRAS_TOKEN:-$GITHUB_TOKEN}"

act workflow_dispatch \
  --workflows .github/workflows/compile-pareto-data.yml \
  --job compile \
  --input accuracy_threshold=60 \
  --secret OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  --secret GITHUB_TOKEN="$GITHUB_TOKEN" \
  --secret MCPTAP_EXTRAS_TOKEN="$MCPTAP_EXTRAS_TOKEN" \
  --var EXCLUDE_PROVIDERS_WITH_PLAN=true
