"""Shared helpers for provider model stats collectors."""

from __future__ import annotations

import json
import pathlib
import re
import shutil
from typing import Any

import tomllib


def parse_base_model(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    parts = value.strip().split("/")
    if len(parts) != 2:
        return None
    provider, model = parts
    if (
        not provider
        or not model
        or "\\" in provider
        or "\\" in model
        or provider in {".", ".."}
        or model in {".", ".."}
    ):
        return None
    return provider, model


def model_id_from_path(path: pathlib.Path, models_dir: pathlib.Path) -> str:
    relative = path.relative_to(models_dir)
    return "/".join((*relative.parent.parts, relative.stem))


def load_provider_models(
    models_dir: pathlib.Path,
) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    by_base_model: dict[str, list[dict[str, str]]] = {}
    errors: list[str] = []
    for path in sorted(models_dir.rglob("*.toml")):
        try:
            document = tomllib.loads(path.read_text(encoding="utf-8"))
            base_model = document.get("base_model")
            if not isinstance(base_model, str) or not base_model:
                continue
            by_base_model.setdefault(base_model, []).append(
                {"path": str(path), "model_id": model_id_from_path(path, models_dir)}
            )
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    return by_base_model, errors


def load_router_providers_from_models_dir_struct(
    routers_dir: pathlib.Path,
    *,
    excluded_provider: str | None = None,
) -> tuple[set[str], list[str]]:
    routers: set[str] = set()
    errors: list[str] = []
    for router_file in sorted(routers_dir.glob("*.toml")):
        try:
            document = tomllib.loads(router_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{router_file}: {exc}")
            continue
        if router_file.stem != excluded_provider and document.get("router_providers_from_models_dir_struct") is True:
            routers.add(router_file.stem)
    return routers, errors


def load_routers(routers_dir: pathlib.Path) -> tuple[set[str], list[str]]:
    routers: set[str] = set()
    errors: list[str] = []
    for router_file in sorted(routers_dir.glob("*.toml")):
        try:
            document = tomllib.loads(router_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{router_file}: {exc}")
            continue
        if router_file.stem != "openrouter" and document.get("is_router") is True:
            routers.add(router_file.stem)
    return routers, errors


def router_slug_matches(left: str, right: str) -> bool:
    normalized_left = re.sub(r"[^a-z0-9]+", "", left.lower())
    normalized_right = re.sub(r"[^a-z0-9]+", "", right.lower())
    return bool(normalized_left) and normalized_left == normalized_right


def safe_relative_path(value: str) -> pathlib.Path:
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative output path: {value}")
    if any("\\" in part for part in path.parts):
        raise ValueError(f"unsafe relative output path: {value}")
    return pathlib.Path(*path.parts)


def write_outputs(
    outputs: dict[str, dict[str, Any]],
    output_dir: pathlib.Path,
    *,
    dry_run: bool,
) -> int:
    written = 0
    if dry_run:
        return len(outputs)
    for relative_path, payload in outputs.items():
        destination = output_dir / safe_relative_path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written += 1
    return written


def write_synthetic_stats(
    *,
    providers_dir: pathlib.Path,
    routers_dir: pathlib.Path,
    stats_dir: pathlib.Path,
    synthetic_dir: pathlib.Path,
    dry_run: bool,
    excluded_provider: str | None = None,
) -> tuple[int, list[str], list[str]]:
    structural_routers, errors = load_router_providers_from_models_dir_struct(
        routers_dir,
        excluded_provider=excluded_provider,
    )
    routers, router_errors = load_routers(routers_dir)
    errors.extend(router_errors)
    collisions: list[str] = []
    written = 0
    if not stats_dir.is_dir() or not providers_dir.is_dir():
        return written, errors, collisions
    if not dry_run and synthetic_dir.exists():
        if synthetic_dir.is_dir():
            shutil.rmtree(synthetic_dir)
        else:
            synthetic_dir.unlink()
    stats_root = stats_dir.resolve()
    stats_provider_models_roots: list[tuple[pathlib.Path, pathlib.Path]] = []

    for stats_provider_dir in sorted(path for path in stats_dir.iterdir() if path.is_dir()):
        try:
            stats_provider_root = stats_provider_dir.resolve(strict=True)
        except OSError:
            continue
        if not stats_provider_root.is_relative_to(stats_root):
            continue
        models_root = stats_provider_dir / "models"
        try:
            resolved_models_root = models_root.resolve(strict=True)
        except OSError:
            continue
        if not resolved_models_root.is_relative_to(stats_provider_root):
            continue
        stats_provider_models_roots.append((stats_provider_dir, resolved_models_root))

    for provider_dir in sorted(path for path in providers_dir.iterdir() if path.is_dir()):
        matching_structural_routers = sorted(
            router for router in structural_routers if router_slug_matches(router, provider_dir.name)
        )
        if len(matching_structural_routers) != 1:
            if len(matching_structural_routers) > 1:
                errors.append(
                    f"{provider_dir}: matches multiple structured routers: {', '.join(matching_structural_routers)}"
                )
            continue
        matched_router = matching_structural_routers[0]
        models_dir = provider_dir / "models"
        if not models_dir.is_dir():
            continue

        for model_file in sorted(models_dir.rglob("*.toml")):
            try:
                model_document = tomllib.loads(model_file.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
                errors.append(f"{model_file}: {exc}")
                continue
            parsed_base_model = parse_base_model(model_document.get("base_model"))
            if parsed_base_model is None:
                continue
            base_model_provider, base_model_slug = parsed_base_model
            model_relative_path = model_file.relative_to(models_dir)
            source_candidates: dict[pathlib.Path, pathlib.Path] = {}
            if matched_router in routers:
                matching_stats_provider_models_roots = [
                    (stats_provider_dir, resolved_models_root)
                    for stats_provider_dir, resolved_models_root in stats_provider_models_roots
                    if any(
                        router_slug_matches(component, stats_provider_dir.name)
                        for component in model_relative_path.parent.parts
                    )
                ]
            else:
                matching_stats_provider_models_roots = [
                    (stats_provider_dir, resolved_models_root)
                    for stats_provider_dir, resolved_models_root in stats_provider_models_roots
                    if router_slug_matches(provider_dir.name, stats_provider_dir.name)
                ]

            for stats_provider_dir, resolved_models_root in matching_stats_provider_models_roots:
                source = stats_provider_dir / "models" / base_model_provider / f"{base_model_slug}.json"
                if not source.is_file():
                    continue
                try:
                    resolved_source = source.resolve(strict=True)
                except OSError:
                    continue
                if resolved_source.is_relative_to(resolved_models_root):
                    source_candidates[resolved_source] = source

            if not source_candidates:
                continue
            destination = synthetic_dir / provider_dir.name / "models" / model_relative_path.with_suffix(".json")
            if len(source_candidates) > 1:
                sources = ", ".join(str(source.relative_to(stats_dir)) for source in sorted(source_candidates.values()))
                collisions.append(f"{destination.relative_to(synthetic_dir)}: {sources}")
                continue
            resolved_source = next(iter(source_candidates))
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(resolved_source.read_bytes())
            written += 1

    return written, errors, collisions
