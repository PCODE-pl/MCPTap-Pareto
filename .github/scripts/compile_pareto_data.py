#!/usr/bin/env python3
"""Compile OpenRouter GPQA coding results with provider pricing from TOML files."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import tomllib

DEFAULT_BENCHMARK_URL = (
    "https://openrouter.ai/api/v1/benchmarks?benchmark_type=gpqa_diamond&source=artificial-analysis&task_type=coding"
)
DEFAULT_ACCURACY_THRESHOLD = 60
DEFAULT_EXCLUDE_PROVIDERS_WITH_PLAN = True
DATE_SUFFIX_RE = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2})$")


def parse_bool(value: str | None, default: bool) -> bool:
    """Parse a boolean environment value with an explicit default."""
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def normalize_model_ref(value: str) -> str:
    """Normalize a model reference for deterministic alias matching."""
    value = value.strip().lower().lstrip("~")
    return DATE_SUFFIX_RE.sub("", value)


def normalize_display_name(value: str) -> str:
    """Normalize display names while ignoring lab prefixes and suffix aliases."""
    value = value.split(":", 1)[-1]
    value = re.sub(r"\b(?:latest|preview)\b", "", value, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def slugify_display_name(value: str) -> str:
    """Convert a benchmark display name into a model-reference slug."""
    while True:
        stripped = re.sub(r"\s*\([^()]*\)", "", value)
        if stripped == value:
            break
        value = stripped
    value = value.split(":", 1)[-1].strip().lower()
    return re.sub(r"[^a-z0-9.]+", "-", value).strip("-")


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


def load_average_stats(repo_root: Path, provider_name: str, model_id: str) -> dict[str, Any]:
    """Load average statistics for one exact provider/model path."""
    stats_path = repo_root / "stats" / "_average" / provider_name / "models" / f"{model_id}.json"
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
        raise TypeError("OpenRouter API response does not contain a data list")
    return payload


def build_canonical_metadata(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Build canonical model metadata mappings from models/*.toml."""
    metadata: dict[str, dict[str, Any]] = {}
    models_root = repo_root / "models"
    for path in models_root.rglob("*.toml"):
        model_id = path.relative_to(models_root).with_suffix("").as_posix()
        data = load_toml(path)
        if data and isinstance(data.get("name"), str):
            metadata[model_id] = data
    return metadata


def build_openrouter_refs(repo_root: Path) -> list[dict[str, Any]]:
    """Return OpenRouter model files and their canonical/family metadata."""
    refs: list[dict[str, Any]] = []
    models_root = repo_root / "providers" / "openrouter" / "models"
    for path in models_root.rglob("*.toml"):
        model_id = path.relative_to(models_root).with_suffix("").as_posix()
        data = load_toml(path)
        if not data:
            continue
        base_model = data.get("base_model")
        refs.append(
            {
                "model_id": model_id,
                "base_model": base_model.strip() if isinstance(base_model, str) else None,
                "family": data.get("family") if isinstance(data.get("family"), str) else None,
                "name": data.get("name") if isinstance(data.get("name"), str) else None,
                "reasoning": bool(data.get("reasoning")),
            }
        )
    return refs


def resolve_canonical_model(
    benchmark_record: dict[str, Any],
    openrouter_refs: list[dict[str, Any]],
    canonical_metadata: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    """Map one benchmark result to a canonical model and OpenRouter metadata."""
    benchmark_ref = benchmark_record.get("model_permaslug")
    if not isinstance(benchmark_ref, str) or not benchmark_ref.strip():
        return None

    normalized_ref = normalize_model_ref(benchmark_ref)
    exact_refs = [
        ref
        for ref in openrouter_refs
        if normalized_ref
        in {
            normalize_model_ref(ref["model_id"]),
            normalize_model_ref(ref["base_model"] or ""),
        }
    ]
    if len(exact_refs) == 1:
        ref = exact_refs[0]
        return ref["base_model"] or ref["model_id"], ref
    if len(exact_refs) > 1:
        canonical_models = {ref["base_model"] or ref["model_id"] for ref in exact_refs}
        if len(canonical_models) == 1:
            ref = exact_refs[0]
            return canonical_models.pop(), ref
        raise RuntimeError(f"Ambiguous OpenRouter model mapping for {benchmark_ref}: {sorted(canonical_models)}")

    display_name = benchmark_record.get("display_name")
    if isinstance(display_name, str):
        display_key = normalize_display_name(display_name)
        named_refs = [
            ref for ref in openrouter_refs if ref["name"] and normalize_display_name(ref["name"]) == display_key
        ]
        if len(named_refs) == 1:
            ref = named_refs[0]
            return ref["base_model"] or ref["model_id"], ref

        name_matches = {
            model_id
            for model_id, metadata in canonical_metadata.items()
            if normalize_display_name(metadata["name"]) == display_key
            and any(ref["base_model"] == model_id or ref["model_id"] == model_id for ref in openrouter_refs)
        }
        if len(name_matches) == 1:
            canonical_model = name_matches.pop()
            matching_refs = [
                ref
                for ref in openrouter_refs
                if ref["base_model"] == canonical_model or ref["model_id"] == canonical_model
            ]
            return canonical_model, matching_refs[0]
        if len(name_matches) > 1:
            raise RuntimeError(f"Ambiguous display-name mapping for {display_name}: {sorted(name_matches)}")

        heuristic_slug = slugify_display_name(display_name)
        heuristic_refs = [
            ref
            for ref in openrouter_refs
            if heuristic_slug
            in {
                normalize_model_ref(ref["model_id"]).rsplit("/", 1)[-1],
                normalize_model_ref(ref["base_model"] or "").rsplit("/", 1)[-1],
            }
        ]
        if len(heuristic_refs) == 1:
            ref = heuristic_refs[0]
            print(
                f"Using display-name slug mapping for {display_name!r}: {heuristic_slug}",
                file=sys.stderr,
            )
            return ref["base_model"] or ref["model_id"], ref
        if len(heuristic_refs) > 1:
            canonical_models = {ref["base_model"] or ref["model_id"] for ref in heuristic_refs}
            if len(canonical_models) == 1:
                ref = heuristic_refs[0]
                print(
                    f"Using display-name slug mapping for {display_name!r}: {heuristic_slug}",
                    file=sys.stderr,
                )
                return canonical_models.pop(), ref
            raise RuntimeError(f"Ambiguous display-name slug mapping for {display_name}: {sorted(canonical_models)}")

    return None


def collect_provider_prices(
    repo_root: Path,
    canonical_model: str,
    canonical_metadata: dict[str, dict[str, Any]],
    openrouter_ref: dict[str, Any],
    exclude_providers_with_plan: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Collect provider records by canonical ID and OpenRouter family fallback."""
    providers: dict[str, dict[str, dict[str, Any]]] = {}
    providers_root = repo_root / "providers"
    canonical_is_thinking = bool(
        canonical_metadata.get(canonical_model, {}).get("reasoning") or openrouter_ref.get("reasoning")
    )
    target_family = openrouter_ref.get("family")
    target_name = openrouter_ref.get("name")
    target_name_key = normalize_display_name(target_name) if target_name else None
    for provider_dir in providers_root.iterdir():
        if exclude_providers_with_plan and "-plan" in provider_dir.name.lower():
            continue
        models_root = provider_dir / "models"
        if not models_root.is_dir():
            continue
        for path in models_root.rglob("*.toml"):
            model_id = path.relative_to(models_root).with_suffix("").as_posix()
            data = load_toml(path)
            if not data or data.get("status") == "alpha":
                continue
            base_model = data.get("base_model")
            direct_match = base_model == canonical_model or model_id == canonical_model
            family_match = (
                not direct_match
                and target_family
                and data.get("family") == target_family
                and (
                    not target_name_key
                    or not isinstance(data.get("name"), str)
                    or normalize_display_name(data["name"]) == target_name_key
                )
            )
            if not direct_match and not family_match:
                continue

            is_thinking = canonical_is_thinking or bool(data.get("reasoning")) or "thinking" in model_id.lower()
            provider_models = providers.setdefault(provider_dir.name, {})
            provider_models[model_id] = {
                "is_thinking": is_thinking,
                "cost": data.get("cost") if isinstance(data.get("cost"), dict) else {},
                "stats": load_average_stats(repo_root, provider_dir.name, model_id),
            }
    return {provider: dict(sorted(models.items())) for provider, models in sorted(providers.items())}


def compile_pareto_data(
    repo_root: Path,
    benchmark_payload: dict[str, Any],
    accuracy_threshold: float,
    exclude_providers_with_plan: bool,
) -> dict[str, dict[str, Any]]:
    """Compile the requested canonical-model-to-provider-price structure."""
    canonical_metadata = build_canonical_metadata(repo_root)
    openrouter_refs = build_openrouter_refs(repo_root)
    result: dict[str, dict[str, Any]] = {}

    for record in benchmark_payload["data"]:
        if not isinstance(record, dict):
            continue
        coding_index = record.get("coding_index")
        if not isinstance(coding_index, (int, float)) or isinstance(coding_index, bool):
            continue
        if coding_index <= accuracy_threshold:
            continue

        resolved = resolve_canonical_model(record, openrouter_refs, canonical_metadata)
        if resolved is None:
            print(
                f"Skipping benchmark model without local OpenRouter mapping: {record.get('model_permaslug')}",
                file=sys.stderr,
            )
            continue

        canonical_model, openrouter_ref = resolved
        result[canonical_model] = {
            "accuracy": coding_index,
            "providers": collect_provider_prices(
                repo_root,
                canonical_model,
                canonical_metadata,
                openrouter_ref,
                exclude_providers_with_plan,
            ),
        }

    return dict(sorted(result.items()))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    threshold = float(os.environ.get("PARETO_ACCURACY_THRESHOLD", DEFAULT_ACCURACY_THRESHOLD))
    exclude_providers_with_plan = parse_bool(
        os.environ.get("EXCLUDE_PROVIDERS_WITH_PLAN"),
        DEFAULT_EXCLUDE_PROVIDERS_WITH_PLAN,
    )
    output_path = Path(os.environ.get("PARETO_OUTPUT_PATH", repo_root / "pareto.json"))
    benchmark_url = os.environ.get("OPENROUTER_BENCHMARK_URL", DEFAULT_BENCHMARK_URL)
    payload = fetch_benchmark(benchmark_url, api_key)
    result = compile_pareto_data(
        repo_root,
        payload,
        threshold,
        exclude_providers_with_plan,
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result)} Pareto models to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
