#!/usr/bin/env python3
"""Build provider model statistics averaged across configured statistics sources."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from dataclasses import dataclass
from numbers import Real
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MIN_VALID_UPTIME = 30

_MISSING = object()


@dataclass(frozen=True)
class MetricSpec:
    """Describe one value read from a source statistics document."""

    source_path: tuple[str, ...]
    output_path: tuple[str, ...]


@dataclass(frozen=True)
class SourceSpec:
    """Describe one statistics source and its supported metrics."""

    name: str
    metrics: tuple[MetricSpec, ...]


SOURCE_SPECS = (
    SourceSpec(
        name="openrouter",
        metrics=(
            MetricSpec(("uptime_last_30m",), ("uptime_last_30m",)),
            MetricSpec(("uptime_last_5m",), ("uptime_last_5m",)),
            MetricSpec(("uptime_last_1d",), ("uptime_last_1d",)),
            MetricSpec(("latency_last_30m", "p50"), ("latency_last_30m",)),
            MetricSpec(("throughput_last_30m", "p50"), ("throughput_last_30m",)),
        ),
    ),
    SourceSpec(
        name="vercel",
        metrics=(
            MetricSpec(("uptime_last_15m",), ("uptime_last_15m",)),
            MetricSpec(("uptime_last_1h",), ("uptime_last_1h",)),
            MetricSpec(("uptime_last_1d",), ("uptime_last_1d",)),
            MetricSpec(("latency_last_1h", "p50"), ("latency_last_1h",)),
            MetricSpec(("throughput_last_1h", "p50"), ("throughput_last_1h",)),
        ),
    ),
    SourceSpec(
        name="llmgateway",
        metrics=(
            MetricSpec(("uptime",), ("uptime_last_1d",)),
            MetricSpec(("avgTimeToFirstToken",), ("latency_last_1d",)),
            MetricSpec(("tokensPerSecond",), ("throughput_last_1d",)),
        ),
    ),
)


def _read_value(document: dict[str, Any], path: tuple[str, ...]) -> object:
    current: object = document
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return _MISSING if current is None else current


def _read_stats_file(path: pathlib.Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{path}: cannot read JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path}: JSON root must be an object"
    return payload, None


def _load_source_document(
    repo_root: pathlib.Path,
    source: SourceSpec,
    relative_model_path: pathlib.Path,
) -> tuple[dict[str, Any] | None, str | None]:
    direct_path = repo_root / "stats" / source.name / relative_model_path
    synthetic_path = repo_root / "stats" / "_synthetic" / source.name / relative_model_path
    for candidate in (direct_path, synthetic_path):
        if candidate.is_file():
            return _read_stats_file(candidate)
    return None, None


def _set_value(document: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    current = document
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value


def _average_values(values: list[object], context: str) -> tuple[object | None, str | None]:
    if len(values) == 1:
        return values[0], None
    if not all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
        return None, f"{context}: cannot average non-numeric values"
    return sum(values) / len(values), None  # type: ignore[arg-type]


def _is_valid_metric(metric: MetricSpec, value: object) -> bool:
    if metric.source_path[0] != "uptime" and not metric.source_path[0].startswith("uptime_last_"):
        return True
    return not (isinstance(value, Real) and not isinstance(value, bool) and value < MIN_VALID_UPTIME)


def _average_model_stats(
    repo_root: pathlib.Path,
    relative_model_path: pathlib.Path,
) -> tuple[dict[str, Any], list[str]]:
    values_by_output_path: dict[tuple[str, ...], list[object]] = {}
    errors: list[str] = []

    for source in SOURCE_SPECS:
        document, error = _load_source_document(repo_root, source, relative_model_path)
        if error is not None:
            errors.append(error)
        if document is None:
            continue
        for metric in source.metrics:
            value = _read_value(document, metric.source_path)
            if value is not _MISSING and _is_valid_metric(metric, value):
                values_by_output_path.setdefault(metric.output_path, []).append(value)

    result: dict[str, Any] = {}
    for output_path, values in values_by_output_path.items():
        value, error = _average_values(values, f"{relative_model_path}: {'.'.join(output_path)}")
        if error is not None:
            errors.append(error)
            continue
        _set_value(result, output_path, value)
    return result, errors


def _model_files(providers_dir: pathlib.Path):
    if not providers_dir.is_dir():
        return
    for provider_dir in sorted(path for path in providers_dir.iterdir() if path.is_dir()):
        models_dir = provider_dir / "models"
        if not models_dir.is_dir():
            continue
        yield from sorted(models_dir.rglob("*.toml"))


def compute_stats(repo_root: pathlib.Path = REPO_ROOT, *, dry_run: bool = False) -> tuple[int, list[str]]:
    """Compute and optionally write average stats for every provider model TOML."""
    providers_dir = repo_root / "providers"
    output_dir = repo_root / "stats" / "_average"
    errors: list[str] = []
    outputs: dict[pathlib.Path, dict[str, Any]] = {}

    if not providers_dir.is_dir():
        errors.append(f"{providers_dir}: providers directory does not exist")
    else:
        for model_file in _model_files(providers_dir):
            relative_model_path = model_file.relative_to(providers_dir).with_suffix(".json")
            payload, model_errors = _average_model_stats(repo_root, relative_model_path)
            errors.extend(model_errors)
            if payload:
                outputs[relative_model_path] = payload

    if not dry_run:
        if output_dir.exists():
            if output_dir.is_dir() and not output_dir.is_symlink():
                shutil.rmtree(output_dir)
            else:
                output_dir.unlink()
        for relative_path, payload in sorted(outputs.items()):
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return len(outputs), errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute statistics without writing or removing JSON files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written, errors = compute_stats(dry_run=args.dry_run)
    print("Average provider model stats")
    print("============================")
    print(f"Models with source statistics: {written}")
    if args.dry_run:
        print("Output files: would write")
    else:
        print(f"Output files: {written}")
    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors:
            print(f"  - {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
