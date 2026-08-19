#!/bin/bash

set -euo pipefail

act workflow_dispatch \
  --workflows .github/workflows/compile-pareto-data.yml \
  --job compile \
  --input accuracy_threshold=60 \
  --secret OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  --var EXCLUDE_PROVIDERS_WITH_PLAN=true
