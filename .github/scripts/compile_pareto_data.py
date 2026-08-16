#!/usr/bin/env python3
"""Compile OpenRouter tau-bench results with provider pricing from TOML files."""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BENCHMARK_URL = (
    "https://openrouter.ai/api/v1/benchmarks"
    "?source=openrouter&benchmark_type=tau_bench_verified_airline"
)
DEFAULT_ACCURACY_THRESHOLD = 0.60
DATE_SUFFIX_RE = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2})$")


def normalize_model_ref(value: str) -> str:
    """Normalize a model reference for deterministic alias matching."""
    value = value.strip().lower().lstrip("~")
    return DATE_SUFFIX_RE.sub("", value)


def normalize_display_name(value: str) -> str:
    """Normalize display names while ignoring lab prefixes and suffix aliases."""
    value = value.split(":", 1)[-1]
    value = re.sub(r"\b(?:latest|preview)\b", "", value, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_toml(path: Path) -> dict[str, Any] | None:
    """Read a TOML file, ignoring broken symlinks and malformed optional files."""
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except OSError as exc:
        print(f"Skipping unreadable TOML file {path}: {exc}", file=sys.stderr)
        return None
    except tomllib.TOMLDecodeError as exc:
        print(f"Skipping malformed TOML file {path}: {exc}", file=sys.stderr)
        return None


def fetch_benchmark(url: str, api_key: str) -> dict[str, Any]:
    """Fetch the authenticated OpenRouter benchmark response."""
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"OpenRouter API returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter API request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenRouter API returned invalid JSON") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError("OpenRouter API response does not contain a data list")
    return payload


def build_canonical_names(repo_root: Path) -> dict[str, str]:
    """Build canonical model ID to display-name mappings from models/*.toml."""
    names: dict[str, str] = {}
    models_root = repo_root / "models"
    for path in models_root.rglob("*.toml"):
        model_id = path.relative_to(models_root).with_suffix("").as_posix()
        data = load_toml(path)
        if data and isinstance(data.get("name"), str):
            names[model_id] = data["name"]
    return names


def build_openrouter_refs(repo_root: Path) -> list[tuple[str, str]]:
    """Return OpenRouter provider model IDs and their canonical base models."""
    refs: list[tuple[str, str]] = []
    models_root = repo_root / "providers" / "openrouter" / "models"
    for path in models_root.rglob("*.toml"):
        model_id = path.relative_to(models_root).with_suffix("").as_posix()
        data = load_toml(path)
        if not data:
            continue
        base_model = data.get("base_model")
        if isinstance(base_model, str) and base_model.strip():
            refs.append((model_id, base_model.strip()))
    return refs


def resolve_canonical_model(
    benchmark_record: dict[str, Any],
    openrouter_refs: list[tuple[str, str]],
    canonical_names: dict[str, str],
) -> str | None:
    """Map one benchmark result to a canonical model ID."""
    benchmark_ref = benchmark_record.get("model_permaslug")
    if not isinstance(benchmark_ref, str) or not benchmark_ref.strip():
        return None

    normalized_ref = normalize_model_ref(benchmark_ref)
    exact_matches = {
        base_model
        for provider_model, base_model in openrouter_refs
        if normalized_ref in {
            normalize_model_ref(provider_model),
            normalize_model_ref(base_model),
        }
    }
    if len(exact_matches) == 1:
        return exact_matches.pop()
    if len(exact_matches) > 1:
        raise RuntimeError(f"Ambiguous OpenRouter model mapping for {benchmark_ref}: {sorted(exact_matches)}")

    display_name = benchmark_record.get("display_name")
    if isinstance(display_name, str):
        display_key = normalize_display_name(display_name)
        name_matches = {
            model_id
            for model_id, name in canonical_names.items()
            if normalize_display_name(name) == display_key
            and any(base_model == model_id for _, base_model in openrouter_refs)
        }
        if len(name_matches) == 1:
            return name_matches.pop()
        if len(name_matches) > 1:
            raise RuntimeError(f"Ambiguous display-name mapping for {display_name}: {sorted(name_matches)}")

    return None


def collect_provider_prices(repo_root: Path, canonical_model: str) -> dict[str, dict[str, Any]]:
    """Collect all provider TOML records serving the canonical model."""
    providers: dict[str, dict[str, Any]] = {}
    providers_root = repo_root / "providers"
    for provider_dir in providers_root.iterdir():
        models_root = provider_dir / "models"
        if not models_root.is_dir():
            continue
        for path in models_root.rglob("*.toml"):
            model_id = path.relative_to(models_root).with_suffix("").as_posix()
            data = load_toml(path)
            if not data or data.get("status") == "alpha":
                continue
            base_model = data.get("base_model")
            if base_model != canonical_model and model_id != canonical_model:
                continue

            provider_key = provider_dir.name
            if provider_key in providers:
                provider_key = f"{provider_key}/{model_id}"
                suffix = 2
                while provider_key in providers:
                    provider_key = f"{provider_dir.name}/{model_id}#{suffix}"
                    suffix += 1

            cost = data.get("cost")
            providers[provider_key] = {
                "model": model_id,
                "cost": cost if isinstance(cost, dict) else {},
            }
    return dict(sorted(providers.items()))


def compile_pareto_data(
    repo_root: Path,
    benchmark_payload: dict[str, Any],
    accuracy_threshold: float,
) -> dict[str, dict[str, Any]]:
    """Compile the requested canonical-model-to-provider-price structure."""
    canonical_names = build_canonical_names(repo_root)
    openrouter_refs = build_openrouter_refs(repo_root)
    result: dict[str, dict[str, Any]] = {}

    for record in benchmark_payload["data"]:
        if not isinstance(record, dict):
            continue
        accuracy = record.get("accuracy")
        if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
            continue
        if accuracy <= accuracy_threshold:
            continue

        canonical_model = resolve_canonical_model(record, openrouter_refs, canonical_names)
        if canonical_model is None:
            print(
                f"Skipping benchmark model without local OpenRouter mapping: {record.get('model_permaslug')}",
                file=sys.stderr,
            )
            continue

        result[canonical_model] = {
            "accuracy": accuracy,
            "providers": collect_provider_prices(repo_root, canonical_model),
        }

    return dict(sorted(result.items()))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    threshold = float(os.environ.get("PARETO_ACCURACY_THRESHOLD", DEFAULT_ACCURACY_THRESHOLD))
    output_path = Path(os.environ.get("PARETO_OUTPUT_PATH", repo_root / "pareto.json"))
    benchmark_url = os.environ.get("OPENROUTER_BENCHMARK_URL", DEFAULT_BENCHMARK_URL)
    payload = fetch_benchmark(benchmark_url, api_key)
    result = compile_pareto_data(repo_root, payload, threshold)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result)} Pareto models to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
