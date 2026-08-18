#!/usr/bin/env python3
"""Collect Vercel AI Gateway endpoint statistics for provider-backed models."""

from __future__ import annotations

import argparse
import json
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
    load_provider_models,
    parse_base_model,
    safe_relative_path,
    write_outputs,
    write_synthetic_stats,
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
        "--provider-dir",
        type=pathlib.Path,
        default=repo_root / "providers",
        help="Root directory containing provider model TOMLs for synthetic stats",
    )
    parser.add_argument(
        "--routers-dir",
        type=pathlib.Path,
        default=repo_root / "routers",
        help="Directory containing router TOML files for synthetic stats",
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
        "--model",
        action="append",
        dest="models",
        help="Process only this source or canonical model ID; may be repeated",
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

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Vercel endpoint request failed for {model_id}: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Vercel endpoint request failed for {model_id}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Vercel endpoint returned invalid JSON for {model_id}") from exc
    return parse_vercel_response(payload, model_id)


def extract_stats(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: endpoint[key] for key in VERCEL_STATS_KEYS if key in endpoint}


def build_outputs(
    models: dict[str, list[dict[str, str]]],
    endpoint_cache: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    outputs: dict[str, dict[str, Any]] = {}
    collisions: list[str] = []
    unmatched_providers: list[str] = []

    for canonical_model, entries in sorted(models.items()):
        source_entry = next(
            (entry for entry in entries if entry["model_id"] == canonical_model),
            min(entries, key=lambda entry: entry["model_id"]),
        )
        source_model_id = source_entry["model_id"]
        parsed_model = parse_base_model(canonical_model)
        if parsed_model is None:
            unmatched_providers.append(f"{canonical_model}: invalid base_model")
            continue
        model_provider, model_name = parsed_model

        for endpoint in endpoint_cache.get(source_model_id, []):
            provider_name = endpoint.get("provider_name")
            if not isinstance(provider_name, str) or not provider_name.strip():
                unmatched_providers.append(f"{source_model_id}: endpoint has no provider_name")
                continue
            try:
                output_key = safe_relative_path(f"{provider_name}/models/{model_provider}/{model_name}.json").as_posix()
            except ValueError:
                unmatched_providers.append(f"{source_model_id}: unsafe provider_name {provider_name!r}")
                continue
            if output_key in outputs:
                collisions.append(output_key)
                continue
            outputs[output_key] = extract_stats(endpoint)

    return outputs, sorted(set(collisions)), sorted(set(unmatched_providers))


def print_bucket(title: str, items: list[str], limit: int = 100) -> None:
    print(f"\n{title} ({len(items)})")
    if not items:
        print("  - none")
        return
    for item in items[:limit]:
        print(f"  - {item}")
    if len(items) > limit:
        print(f"  ... and {len(items) - limit} more")


def main() -> int:
    args = parse_args()
    models, parse_errors = load_provider_models(args.models_dir)
    if args.models:
        selected = set(args.models)
        models = {
            canonical_model: entries
            for canonical_model, entries in models.items()
            if canonical_model in selected or any(entry["model_id"] in selected for entry in entries)
        }

    source_model_ids = sorted({entry["model_id"] for entries in models.values() for entry in entries})
    endpoint_cache: dict[str, list[dict[str, Any]]] = {}
    api_errors: list[str] = []
    for model_id in source_model_ids:
        try:
            endpoint_cache[model_id] = fetch_model_endpoints(args.api_url, model_id)
        except RuntimeError as exc:
            api_errors.append(str(exc))

    outputs, output_collisions, unmatched_providers = build_outputs(models, endpoint_cache)
    written = write_outputs(outputs, args.output_dir, dry_run=args.dry_run)
    synthetic_written, synthetic_errors, synthetic_collisions = write_synthetic_stats(
        providers_dir=args.provider_dir,
        routers_dir=args.routers_dir,
        stats_dir=args.output_dir,
        synthetic_dir=args.synthetic_output_dir,
        dry_run=args.dry_run,
        excluded_provider="vercel",
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
