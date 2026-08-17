#!/usr/bin/env python3
"""Compare NanoGPT provider TOML prices with the current NanoGPT API."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from typing import Any

import tomllib

DEFAULT_API_URL = "https://nano-gpt.com/api/v1/models?detailed=true"
PRICE_FIELDS = ("input", "output", "cache_read", "cache_write")
PRICE_TOLERANCE = Decimal("0.000001")
ORG_NORMALIZATION = {
    "nousresearch": "NousResearch",
    "qwen": "qwen",
    "thedrummer": "TheDrummer",
}


def parse_args() -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default=os.environ.get("NANOGPT_MODELS_URL", DEFAULT_API_URL),
        help="NanoGPT detailed models API URL",
    )
    parser.add_argument(
        "--provider-dir",
        type=pathlib.Path,
        default=repo_root / "providers" / "nano-gpt" / "models",
        help="Directory containing NanoGPT provider TOMLs",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when a price mismatch or missing API model is found",
    )
    return parser.parse_args()


def fetch_models(api_url: str) -> dict[str, dict[str, Any]]:
    api_key = os.environ.get("NANOGPT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NANOGPT_API_KEY is required")

    request = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"NanoGPT API returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"NanoGPT API request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("NanoGPT API returned invalid JSON") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise TypeError("NanoGPT API response does not contain a data list")

    models: dict[str, dict[str, Any]] = {}
    for record in payload["data"]:
        if isinstance(record, dict) and isinstance(record.get("id"), str):
            models[normalize_model_id(record["id"])] = record
    return models


def normalize_model_id(model_id: str) -> str:
    parts = model_id.split("/")
    if len(parts) < 2:
        return model_id
    normalized_org = ORG_NORMALIZATION.get(parts[0].lower(), parts[0])
    return "/".join([normalized_org, *parts[1:]])


def model_id_from_path(path: pathlib.Path, provider_dir: pathlib.Path) -> str:
    relative = path.relative_to(provider_dir)
    filename = relative.name
    if not filename.endswith(".toml"):
        raise ValueError(f"Not a TOML file: {path}")
    model_name = filename[: -len(".toml")]
    if relative.parent == pathlib.Path("."):
        return model_name
    return "/".join((*relative.parent.parts, model_name))


def as_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def local_price(value: Any) -> Decimal | None:
    return as_decimal(value)


def api_price_per_million(value: Any, unit: Any = "per_million_tokens") -> Decimal | None:
    price = as_decimal(value)
    if price is None:
        return None
    if unit in {"per_token", "per-token"}:
        return price * Decimal(1000000)
    if unit in {"per_1k_tokens", "per_1k", "per-1k-tokens"}:
        return price * Decimal(1000)
    return price


def normalize_local_tiers(value: Any) -> list[dict[str, Decimal | int | None]]:
    if not isinstance(value, list):
        return []
    tiers: list[dict[str, Decimal | int | None]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        tier = item.get("tier")
        size = tier.get("size") if isinstance(tier, dict) else None
        if not isinstance(size, int):
            continue
        tiers.append(
            {
                "size": size,
                **{field: local_price(item.get(field)) for field in PRICE_FIELDS},
            }
        )
    return tiers


def normalize_local_cost(cost: Any) -> dict[str, Any] | None:
    if not isinstance(cost, dict):
        return None
    return {
        **{field: local_price(cost.get(field)) for field in PRICE_FIELDS},
        "tiers": normalize_local_tiers(cost.get("tiers")),
    }


def normalize_remote_cost(record: dict[str, Any]) -> dict[str, Any] | None:
    pricing = record.get("pricing")
    if not isinstance(pricing, dict):
        return None
    unit = pricing.get("unit", "per_million_tokens")
    return {
        "input": api_price_per_million(pricing.get("input", pricing.get("prompt")), unit),
        "output": api_price_per_million(pricing.get("output", pricing.get("completion")), unit),
        "cache_read": api_price_per_million(pricing.get("cacheReadInputPer1kTokens"), "per_1k_tokens"),
        "cache_write": api_price_per_million(pricing.get("cacheWriteInputPer1kTokens"), "per_1k_tokens"),
        "tiers": [],
    }


def format_decimal(value: Decimal | None) -> str:
    if value is None:
        return "missing"
    return format(value, "f")


def prices_equal(field: str, local: Decimal | None, remote: Decimal | None) -> bool:
    if local is None and remote is None:
        return True
    if field in {"cache_read", "cache_write"} and {local, remote} == {
        None,
        Decimal(0),
    }:
        return True
    if local is None or remote is None:
        return False
    return abs(local - remote) <= PRICE_TOLERANCE


def format_tiers(value: Any) -> str:
    if not isinstance(value, list):
        return "[]"
    rendered = []
    for tier in value:
        if not isinstance(tier, dict):
            rendered.append(str(tier))
            continue
        size = tier.get("size", "?")
        prices = ", ".join(f"{field}={format_decimal(tier.get(field))}" for field in PRICE_FIELDS)
        rendered.append(f"size={size} ({prices})")
    return "[" + "; ".join(rendered) + "]"


def compare_costs(
    local: dict[str, Any] | None,
    remote: dict[str, Any] | None,
) -> list[str]:
    if local is None:
        return ["local cost section is missing"]
    if remote is None:
        return ["remote pricing section is missing"]

    differences: list[str] = []
    for field in PRICE_FIELDS:
        local_value = local.get(field)
        remote_value = remote.get(field)
        if not prices_equal(field, local_value, remote_value):
            differences.append(f"{field}: local={format_decimal(local_value)} remote={format_decimal(remote_value)}")

    if local.get("tiers", []) != remote.get("tiers", []):
        differences.append(
            f"tiers: local={format_tiers(local.get('tiers'))} remote={format_tiers(remote.get('tiers'))}"
        )
    return differences


def load_local_models(
    provider_dir: pathlib.Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    models: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    if not provider_dir.is_dir():
        raise RuntimeError(f"Provider directory does not exist: {provider_dir}")

    for path in sorted(provider_dir.rglob("*.toml")):
        try:
            model_id = normalize_model_id(model_id_from_path(path, provider_dir))
            document = tomllib.loads(path.read_text(encoding="utf-8"))
            models[model_id] = {
                "path": path,
                "cost": normalize_local_cost(document.get("cost")),
            }
        except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    return models, errors


def print_bucket(title: str, items: list[str], limit: int = 100) -> None:
    print(f"\n{title} ({len(items)})")
    if not items:
        print("  - none")
        return
    for item in items[:limit]:
        print(f"  - {item}")
    if len(items) > limit:
        print(f"  ... and {len(items) - limit} more")


def verify(provider_dir: pathlib.Path, api_url: str) -> dict[str, Any]:
    local_models, parse_errors = load_local_models(provider_dir)
    remote_models = fetch_models(api_url)

    matches = 0
    mismatches: list[str] = []
    missing_remote: list[str] = []
    for model_id, local in local_models.items():
        remote = remote_models.get(model_id)
        if remote is None:
            missing_remote.append(model_id)
            continue
        differences = compare_costs(local["cost"], normalize_remote_cost(remote))
        if differences:
            mismatches.append(f"{model_id}: {'; '.join(differences)}")
        else:
            matches += 1

    return {
        "local_models": len(local_models),
        "remote_models": len(remote_models),
        "matches": matches,
        "mismatches": mismatches,
        "missing_remote": missing_remote,
        "remote_without_local": sorted(set(remote_models) - set(local_models)),
        "parse_errors": parse_errors,
    }


def main() -> int:
    args = parse_args()
    summary = verify(args.provider_dir, args.api_url)

    print("NanoGPT price verification")
    print("==========================")
    print(f"Local NanoGPT TOMLs: {summary['local_models']}")
    print(f"Models from NanoGPT API: {summary['remote_models']}")
    print(f"Price matches: {summary['matches']}")
    print_bucket("Price mismatches", summary["mismatches"])
    print_bucket("Local model missing from API", summary["missing_remote"])
    print_bucket(
        "API models without local TOML",
        summary["remote_without_local"],
        limit=50,
    )
    print_bucket("TOML parse errors", summary["parse_errors"])

    has_errors = bool(summary["mismatches"] or summary["missing_remote"] or summary["parse_errors"])
    return 1 if args.strict and has_errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
