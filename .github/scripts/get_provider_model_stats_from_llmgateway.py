#!/usr/bin/env python3
"""Collect LLM Gateway backend statistics for provider-backed model TOMLs."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import urllib.parse
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.provider_model_stats import (  # noqa: E402
    DEFAULT_AI_MODEL,
    DEFAULT_PROVIDER_DIR,
    DEFAULT_ROUTERS_DIR,
    build_provider_outputs,
    collect_model_endpoints,
    fetch_json,
    load_mapping_cache,
    load_provider_models,
    load_providers,
    print_bucket,
    query_provider_mappings,
    resolve_provider_deterministically,
    save_mapping_cache,
    should_run_ai_query,
    write_collected_outputs,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_URL = os.environ.get("LLMGATEWAY_STATS_BASE_URL", "https://internal.llmgateway.io").rstrip("/")
MODELS_DIR = REPO_ROOT / "providers" / "llmgateway" / "models"
OUTPUT_DIR = REPO_ROOT / "stats" / "llmgateway"
SYNTHETIC_OUTPUT_DIR = REPO_ROOT / "stats" / "_synthetic" / "llmgateway"
CACHE_FILE_NAME = ".llmgateway_provider_mapping_cache.json"
# Space per-model benchmark requests so the backend is not overwhelmed.
REQUEST_INTERVAL = float(os.environ.get("LLMGATEWAY_STATS_REQUEST_INTERVAL", "0.5"))
PUBLIC_INDEX_RETRIES = 4
PUBLIC_INDEX_RETRY_DELAY = 5
STATS_KEYS = (
    "logsCount",
    "avgTimeToFirstToken",
    "tokensPerSecond",
    "uptime",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ai-model",
        default=os.environ.get("OPENROUTER_AI_MODEL", DEFAULT_AI_MODEL),
        help="OpenRouter model used for provider-name mapping",
    )
    parser.add_argument(
        "--disable-ai",
        action="store_true",
        help="Disable AI provider-name mapping",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear provider-name mapping cache before running",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and resolve data without writing JSON files",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the AI request prompts and response to stderr",
    )
    return parser.parse_args()


def parse_public_stats_response(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        raise TypeError("LLM Gateway public stats response is not an object")
    models = payload.get("models")
    if not isinstance(models, list):
        raise TypeError("LLM Gateway public stats response has no models list")
    return sorted(
        {
            model_id.strip()
            for model in models
            if isinstance(model, dict) and isinstance(model_id := model.get("modelId"), str) and model_id.strip()
        }
    )


def fetch_public_model_ids(api_url: str) -> list[str]:
    payload = fetch_json(
        f"{api_url.rstrip('/')}/public/models/stats?window=24h",
        headers={"Accept": "application/json"},
        error_context="LLM Gateway public model stats request failed",
        retries=PUBLIC_INDEX_RETRIES,
        retry_delay=PUBLIC_INDEX_RETRY_DELAY,
    )
    return parse_public_stats_response(payload)


def parse_benchmarks_response(payload: object, model_id: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise TypeError(f"LLM Gateway benchmark response is not an object for {model_id}")
    providers = payload.get("providers")
    if not isinstance(providers, list):
        raise TypeError(f"LLM Gateway benchmark response has no providers list for {model_id}")

    endpoints: list[dict[str, Any]] = []
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_id = provider.get("providerId")
        if not isinstance(provider_id, str) or not provider_id.strip():
            continue
        endpoint = {"provider_name": provider_id.strip()}
        endpoint.update({key: provider[key] for key in STATS_KEYS if key in provider})
        endpoints.append(endpoint)
    return endpoints


def fetch_model_endpoints(api_url: str, model_id: str) -> list[dict[str, Any]]:
    encoded_model_id = urllib.parse.quote(model_id, safe="")
    url = f"{api_url.rstrip('/')}/internal/models/{encoded_model_id}/benchmarks"
    payload = fetch_json(
        url,
        headers={"Accept": "application/json"},
        error_context=f"LLM Gateway benchmark request failed for {model_id}",
        retries=2,
        retry_delay=1,
    )
    return parse_benchmarks_response(payload, model_id)


def extract_stats(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: endpoint[key] for key in STATS_KEYS if key in endpoint and endpoint[key] is not None}


def main() -> int:
    args = parse_args()
    api_url = BASE_URL
    providers = load_providers(REPO_ROOT / DEFAULT_PROVIDER_DIR)
    cache = {} if args.clear_cache else load_mapping_cache(REPO_ROOT, CACHE_FILE_NAME)

    discovered_model_ids = fetch_public_model_ids(api_url)
    local_models, _ = load_provider_models(MODELS_DIR)
    local_model_ids = {entry["model_id"] for entries in local_models.values() for entry in entries}
    selected_model_ids = sorted(set(discovered_model_ids) & local_model_ids)
    unknown_model_ids = sorted(set(discovered_model_ids) - local_model_ids)

    models, parse_errors, endpoint_cache, api_errors, unavailable_models = collect_model_endpoints(
        models_dir=MODELS_DIR,
        api_url=api_url,
        fetch_model_endpoints=fetch_model_endpoints,
        model_ids=selected_model_ids,
        request_interval=REQUEST_INTERVAL,
    )
    source_model_ids = sorted({entry["model_id"] for entries in models.values() for entry in entries})
    provider_names = sorted(
        {
            str(endpoint.get("provider_name"))
            for endpoints in endpoint_cache.values()
            for endpoint in endpoints
            if endpoint.get("provider_name")
        }
    )
    mapping: dict[str, str | None] = {}
    unresolved_names: list[str] = []
    for provider_name in provider_names:
        deterministic = resolve_provider_deterministically(provider_name, providers)
        if deterministic is not None:
            mapping[provider_name] = deterministic
        elif provider_name in cache and cache[provider_name] is not None:
            mapping[provider_name] = cache[provider_name]
        else:
            unresolved_names.append(provider_name)

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if unresolved_names and not args.disable_ai and api_key:
        if should_run_ai_query():
            ai_mapping = query_provider_mappings(
                unresolved_names,
                providers,
                args.ai_model,
                api_key,
                debug=args.debug,
            )
            mapping.update(ai_mapping)
            cache.update(ai_mapping)
    elif unresolved_names and not args.disable_ai:
        print(
            "Warning: OPENROUTER_API_KEY is not set; unresolved provider names will use their original names.",
            file=sys.stderr,
        )

    for provider_name in unresolved_names:
        if mapping.get(provider_name) is None:
            mapping[provider_name] = provider_name
    cache.update(mapping)

    outputs, output_collisions, unmatched_providers, multiple_endpoints = build_provider_outputs(
        models,
        endpoint_cache,
        resolve_provider=mapping.get,
        extract_stats=extract_stats,
    )

    print("LLM Gateway provider model stats")
    print("=================================")
    print(f"LLM Gateway models with public stats: {len(discovered_model_ids)}")
    print(f"LLM Gateway models matched to TOMLs: {len(selected_model_ids)}")
    print(f"LLM Gateway TOMLs with base_model: {sum(len(v) for v in models.values())}")
    print(f"Unique source model IDs: {len(source_model_ids)}")
    print(f"Provider definitions: {len(providers)}")
    print(f"Provider names from benchmarks: {len(provider_names)}")
    print(f"Cached provider mappings: {len(cache)}")
    print(f"Benchmark responses: {len(endpoint_cache)}")
    print(f"Resolved output files: {len(outputs)}")

    print_bucket("Public models without matching TOML", unknown_model_ids)
    print_bucket("Unmatched provider names", sorted(set(unmatched_providers)))
    print_bucket("Multiple endpoints for provider/model", sorted(set(multiple_endpoints)))
    print_bucket("Output collisions", sorted(set(output_collisions)))
    print_bucket("Unavailable models (HTTP 404)", sorted(set(unavailable_models)))
    print_bucket("API errors", sorted(set(api_errors)))
    print_bucket("TOML parse errors", parse_errors)

    if api_errors or parse_errors:
        print("No output files were changed because collection errors were detected.", file=sys.stderr)
        return 1

    if not args.dry_run and cache:
        save_mapping_cache(REPO_ROOT, cache, CACHE_FILE_NAME)

    written, synthetic_written, synthetic_errors, synthetic_collisions = write_collected_outputs(
        outputs=outputs,
        output_dir=OUTPUT_DIR,
        providers_dir=REPO_ROOT / DEFAULT_PROVIDER_DIR,
        routers_dir=REPO_ROOT / DEFAULT_ROUTERS_DIR,
        synthetic_dir=SYNTHETIC_OUTPUT_DIR,
        excluded_provider="llmgateway",
        dry_run=args.dry_run,
    )
    print(f"Written: {written}" if not args.dry_run else f"Would write: {len(outputs)}")
    print(
        f"Synthetic written: {synthetic_written}" if not args.dry_run else f"Synthetic would write: {synthetic_written}"
    )
    print_bucket("Synthetic output collisions", synthetic_collisions)
    print_bucket("Synthetic stats errors", synthetic_errors)

    return 1 if synthetic_errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
