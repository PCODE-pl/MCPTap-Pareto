#!/usr/bin/env python3
"""Collect Vercel AI Gateway endpoint statistics for provider-backed models."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.provider_model_stats import (  # noqa: E402 # type: ignore
    DEFAULT_PROVIDER_DIR,
    DEFAULT_ROUTERS_DIR,
    build_provider_outputs,
    collect_model_endpoints,
    fetch_json,
    load_provider_models,  # noqa: F401
    parse_base_model,  # noqa: F401
    print_bucket,
    safe_relative_path,  # noqa: F401
    write_collected_outputs,
    write_outputs,  # noqa: F401
    write_synthetic_stats,  # noqa: F401
)

DEFAULT_API_URL = "https://ai-gateway.vercel.sh/v1/models"
VERCEL_STATS_KEYS = (
    "uptime_last_15m",
    "uptime_last_1h",
    "uptime_last_1d",
    "latency_last_1h",
    "throughput_last_1h",
)


def parse_args() -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default=os.environ.get("VERCEL_AI_GATEWAY_MODELS_URL", DEFAULT_API_URL),
        help="Vercel AI Gateway models endpoint base URL",
    )
    parser.add_argument(
        "--models-dir",
        type=pathlib.Path,
        default=repo_root / "providers" / "vercel" / "models",
        help="Directory containing Vercel provider model TOMLs",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=repo_root / "stats" / "vercel",
        help="Output directory for provider/model JSON files",
    )
    parser.add_argument(
        "--synthetic-output-dir",
        type=pathlib.Path,
        default=repo_root / "stats" / "_synthetic" / "vercel",
        help="Output directory for synthetic provider/model JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and resolve data without writing JSON files",
    )
    return parser.parse_args()


def parse_vercel_response(payload: object, model_id: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise TypeError(f"Vercel endpoint response is not an object for {model_id}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise TypeError(f"Vercel endpoint response has no data object for {model_id}")
    endpoints = data.get("endpoints")
    if not isinstance(endpoints, list):
        raise TypeError(f"Vercel endpoint response has no endpoints list for {model_id}")
    return [endpoint for endpoint in endpoints if isinstance(endpoint, dict)]


def fetch_model_endpoints(api_url: str, model_id: str) -> list[dict[str, Any]]:
    encoded_model_id = urllib.parse.quote(model_id, safe="/")
    url = f"{api_url.rstrip('/')}/{encoded_model_id}/endpoints"
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = fetch_json(
        url,
        headers=headers,
        error_context=f"Vercel endpoint request failed for {model_id}",
    )
    return parse_vercel_response(payload, model_id)


def extract_stats(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: endpoint[key] for key in VERCEL_STATS_KEYS if key in endpoint}


def build_outputs(
    models: dict[str, list[dict[str, str]]],
    endpoint_cache: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    outputs, collisions, unmatched_providers, _ = build_provider_outputs(
        models,
        endpoint_cache,
        resolve_provider=lambda provider_name: provider_name,
        extract_stats=extract_stats,
    )
    return outputs, collisions, unmatched_providers


def main() -> int:
    args = parse_args()
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    models, parse_errors, endpoint_cache, api_errors = collect_model_endpoints(
        models_dir=args.models_dir,
        api_url=args.api_url,
        fetch_model_endpoints=fetch_model_endpoints,
    )
    source_model_ids = sorted({entry["model_id"] for entries in models.values() for entry in entries})

    outputs, output_collisions, unmatched_providers, multiple_endpoints = build_provider_outputs(
        models,
        endpoint_cache,
        resolve_provider=lambda provider_name: provider_name,
        extract_stats=extract_stats,
    )
    written, synthetic_written, synthetic_errors, synthetic_collisions = write_collected_outputs(
        outputs=outputs,
        output_dir=args.output_dir,
        providers_dir=repo_root / DEFAULT_PROVIDER_DIR,
        routers_dir=repo_root / DEFAULT_ROUTERS_DIR,
        synthetic_dir=args.synthetic_output_dir,
        excluded_provider="vercel",
        dry_run=args.dry_run,
    )

    print("Vercel AI Gateway provider model stats")
    print("=======================================")
    print(f"Vercel TOMLs with base_model: {sum(len(entries) for entries in models.values())}")
    print(f"Unique source model IDs: {len(source_model_ids)}")
    print(f"Endpoint responses: {len(endpoint_cache)}")
    print(f"Resolved output files: {len(outputs)}")
    print(f"Written: {written}" if not args.dry_run else f"Would write: {written}")
    print(
        f"Synthetic written: {synthetic_written}" if not args.dry_run else f"Synthetic would write: {synthetic_written}"
    )
    print_bucket("Unmatched endpoints", unmatched_providers)
    print_bucket("Multiple endpoints for provider/model", multiple_endpoints)
    print_bucket("Output collisions", output_collisions)
    print_bucket("API errors", sorted(set(api_errors)))
    print_bucket("TOML parse errors", parse_errors)
    print_bucket("Synthetic output collisions", synthetic_collisions)
    print_bucket("Synthetic stats errors", synthetic_errors)

    return 1 if api_errors or parse_errors or synthetic_errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
