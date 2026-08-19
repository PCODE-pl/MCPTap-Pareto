"""Shared helpers for provider model stats collectors."""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import tomllib

DEFAULT_PROVIDER_MAPPING_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_AI_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_PROVIDER_DIR = "providers"
DEFAULT_ROUTERS_DIR = "routers"


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


def fetch_json(
    url: str,
    *,
    headers: dict[str, str],
    error_context: str,
    timeout: int = 60,
) -> object:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{error_context}: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{error_context}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{error_context}: invalid JSON") from exc


def collect_model_endpoints(
    *,
    models_dir: pathlib.Path,
    api_url: str,
    fetch_model_endpoints: Callable[[str, str], list[dict[str, Any]]],
) -> tuple[
    dict[str, list[dict[str, str]]],
    list[str],
    dict[str, list[dict[str, Any]]],
    list[str],
]:
    models, parse_errors = load_provider_models(models_dir)
    source_model_ids = sorted({entry["model_id"] for entries in models.values() for entry in entries})
    endpoint_cache: dict[str, list[dict[str, Any]]] = {}
    api_errors: list[str] = []
    for model_id in source_model_ids:
        try:
            endpoint_cache[model_id] = fetch_model_endpoints(api_url, model_id)
        except RuntimeError as exc:
            api_errors.append(str(exc))
    return models, parse_errors, endpoint_cache, api_errors


def build_provider_outputs(
    models: dict[str, list[dict[str, str]]],
    endpoint_cache: dict[str, list[dict[str, Any]]],
    *,
    resolve_provider: Callable[[str], str | None],
    extract_stats: Callable[[dict[str, Any]], dict[str, Any]],
    select_endpoint: Callable[[list[dict[str, Any]]], dict[str, Any] | None] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str], list[str]]:
    outputs: dict[str, dict[str, Any]] = {}
    collisions: list[str] = []
    unmatched_providers: list[str] = []
    multiple_endpoints: list[str] = []

    for canonical_model, entries in sorted(models.items()):
        source_entry = next(
            (entry for entry in entries if entry["model_id"] == canonical_model),
            min(entries, key=lambda entry: entry["model_id"]),
        )
        source_model_id = source_entry["model_id"]
        if parse_base_model(canonical_model) is None:
            unmatched_providers.append(f"{canonical_model}: invalid base_model")
            continue
        endpoints_by_provider: dict[str, list[dict[str, Any]]] = {}
        for endpoint in endpoint_cache.get(source_model_id, []):
            provider_name = endpoint.get("provider_name")
            if not isinstance(provider_name, str) or not provider_name.strip():
                unmatched_providers.append(f"{source_model_id}: endpoint has no provider_name")
                continue
            endpoints_by_provider.setdefault(provider_name, []).append(endpoint)

        for provider_name, matched_endpoints in sorted(endpoints_by_provider.items()):
            canonical_provider = resolve_provider(provider_name)
            if canonical_provider is None:
                unmatched_providers.append(provider_name)
                continue
            if len(matched_endpoints) > 1:
                multiple_endpoints.append(
                    f"{provider_name}/{canonical_model}: {len(matched_endpoints)} endpoints; selected deterministic first"
                )
            selected_endpoint = select_endpoint(matched_endpoints) if select_endpoint else matched_endpoints[0]
            if selected_endpoint is None:
                continue
            try:
                output_key = safe_relative_path(f"{canonical_provider}/models/{canonical_model}.json").as_posix()
            except ValueError:
                unmatched_providers.append(f"{provider_name}: unsafe canonical provider {canonical_provider!r}")
                continue
            if output_key in outputs:
                collisions.append(output_key)
                continue
            outputs[output_key] = extract_stats(selected_endpoint)

    return outputs, sorted(set(collisions)), sorted(set(unmatched_providers)), sorted(set(multiple_endpoints))


def print_bucket(title: str, items: list[str], limit: int = 100) -> None:
    print(f"\n{title} ({len(items)})")
    if not items:
        print("  - none")
        return
    for item in items[:limit]:
        print(f"  - {item}")
    if len(items) > limit:
        print(f"  ... and {len(items) - limit} more")


def write_collected_outputs(
    *,
    outputs: dict[str, dict[str, Any]],
    output_dir: pathlib.Path,
    providers_dir: pathlib.Path,
    routers_dir: pathlib.Path,
    synthetic_dir: pathlib.Path,
    excluded_provider: str,
    dry_run: bool,
) -> tuple[int, int, list[str], list[str]]:
    if not dry_run and output_dir.exists():
        if output_dir.is_dir() and not output_dir.is_symlink():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    written = write_outputs(outputs, output_dir, dry_run=dry_run)
    synthetic_written, synthetic_errors, synthetic_collisions = write_synthetic_stats(
        providers_dir=providers_dir,
        routers_dir=routers_dir,
        stats_dir=output_dir,
        synthetic_dir=synthetic_dir,
        dry_run=dry_run,
        excluded_provider=excluded_provider,
    )
    return written, synthetic_written, synthetic_errors, synthetic_collisions


def load_providers(provider_dir: pathlib.Path) -> dict[str, dict[str, str]]:
    providers: dict[str, dict[str, str]] = {}
    for provider_file in sorted(provider_dir.glob("*/provider.toml")):
        try:
            document = tomllib.loads(provider_file.read_text(encoding="utf-8"))
            slug = provider_file.parent.name
            name = document.get("name")
            if isinstance(name, str) and name.strip():
                providers[slug] = {"slug": slug, "name": name.strip()}
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            print(f"Warning: cannot read {provider_file}: {exc}", file=sys.stderr)
    return providers


def load_mapping_cache(repo_root: pathlib.Path, cache_file_name: str) -> dict[str, str | None]:
    path = repo_root / cache_file_name
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(key): value if isinstance(value, str) or value is None else None for key, value in data.items()}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def save_mapping_cache(
    repo_root: pathlib.Path,
    cache: dict[str, str | None],
    cache_file_name: str,
) -> None:
    path = repo_root / cache_file_name
    path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def provider_name_variants(value: str) -> set[str]:
    normalized = normalize_text(value)
    variants = {normalized}
    for suffix in ("ai", "api", "cloud", "llc", "inc", "direct", "router"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
            variants.add(normalized[: -len(suffix)])
    return variants


def resolve_provider_deterministically(
    provider_name: str,
    providers: dict[str, dict[str, str]],
) -> str | None:
    target_variants = provider_name_variants(provider_name)
    matches = []
    for slug, provider in providers.items():
        candidates = provider_name_variants(slug) | provider_name_variants(provider["name"])
        if target_variants & candidates:
            matches.append(slug)

    if len(matches) > 0:
        if "zai" in matches:
            return "zhipuai"
        elif "arcee" in matches:
            return "arcee-ai"
        # elif "stepfun" in matches:
        #     return "stepfun-ai"

    return matches[0] if len(matches) > 0 else None


def query_provider_mappings(
    provider_names: list[str],
    providers: dict[str, dict[str, str]],
    model_name: str,
    api_key: str,
    *,
    debug: bool = False,
) -> dict[str, str | None]:
    if not provider_names:
        return {}

    candidates = [providers[slug] for slug in sorted(providers)]

    system_prompt = (
        "You map OpenRouter provider_name values to canonical provider directory slugs.\n"
        "Return ONLY a JSON object mapping every input provider name to one allowed slug or null.\n"
        "A provider name may omit suffixes such as AI, API, Cloud, Direct, or Router.\n"
        "Prefer the provider.toml name and slug that identify the same company.\n"
        "Never invent a slug: values must be present in ALLOWED_PROVIDERS or null."
    )
    user_prompt = (
        f"ALLOWED_PROVIDERS ({len(candidates)}):\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
        f"PROVIDER_NAMES_TO_MAP ({len(provider_names)}):\n"
        f"{json.dumps(provider_names, ensure_ascii=False)}\n"
    )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    if debug:
        print("AI provider mapping request", file=sys.stderr)
        print(f"URL: {DEFAULT_PROVIDER_MAPPING_CHAT_URL}", file=sys.stderr)
        print(f"Model: {model_name}", file=sys.stderr)
        print("System prompt:", file=sys.stderr)
        print(system_prompt, file=sys.stderr)
        print("User prompt:", file=sys.stderr)
        print(user_prompt, file=sys.stderr)
    request = urllib.request.Request(
        DEFAULT_PROVIDER_MAPPING_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/PCODE-pl/MCPTap-Pareto",
            "X-Title": "MCPTap OpenRouter Provider Stats",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        content = data["choices"][0]["message"]["content"]
        if debug:
            print("AI provider mapping response:", file=sys.stderr)
            print(content, file=sys.stderr)
        parsed = json.loads(content)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: AI provider mapping failed: {exc}", file=sys.stderr)
        return {}

    valid_slugs = set(providers)
    if not isinstance(parsed, dict):
        return {}
    return {
        str(key): value if isinstance(value, str) and value in valid_slugs else None for key, value in parsed.items()
    }


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


def load_routers(
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
        if router_file.stem != excluded_provider and document.get("is_router") is True:
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
    routers, router_errors = load_routers(routers_dir, excluded_provider=excluded_provider)
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
        # print(provider_dir)  # TODO: remove

        matching_structural_routers = sorted(
            router for router in structural_routers if router_slug_matches(router, provider_dir.name)
        )

        # print(matching_structural_routers) # TODO: remove

        if len(matching_structural_routers) != 1:
            if len(matching_structural_routers) > 1:
                errors.append(
                    f"{provider_dir}: matches multiple structured routers: {', '.join(matching_structural_routers)}"
                )
            continue
        matched_router = matching_structural_routers[0]

        if matched_router not in routers:
            continue

        models_dir = provider_dir / "models"
        if not models_dir.is_dir():
            continue

        for model_file in sorted(models_dir.rglob("*.toml")):
            # print(model_file) # TODO: remove
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

            # TODO: remove
            # if matched_router in routers:
            #     # if "arcee" == str(provider_dir.name):
            #     #     print(provider_dir)  # TODO: remove

            #     matching_stats_provider_models_roots = []
            #     for stats_provider_dir, resolved_models_root in stats_provider_models_roots:
            #         if stats_provider_dir.name != base_model_provider:
            #             continue

            #         source = stats_provider_dir / "models" / base_model_provider / f"{base_model_slug}.json"
            #         if source.is_file():
            #             matching_stats_provider_models_roots.append((stats_provider_dir, resolved_models_root))
            #         # else:
            #         #     if "arcee" == str(provider_dir.name):
            #         #         print(f"NOT FOUND: {source}")  # TODO: remove
            # TODO: remove
            # else:
            #     # matching_stats_provider_models_roots = [
            #     #     (stats_provider_dir, resolved_models_root)
            #     #     for stats_provider_dir, resolved_models_root in stats_provider_models_roots
            #     #     if router_slug_matches(provider_dir.name, stats_provider_dir.name)
            #     # ]

            #     matching_stats_provider_models_roots = []
            #     for stats_provider_dir, resolved_models_root in stats_provider_models_roots:
            #         if stats_provider_dir.name != base_model_provider:
            #             continue

            #         source = stats_provider_dir / "models" / base_model_provider / f"{base_model_slug}.json"
            #         if source.is_file():
            #             matching_stats_provider_models_roots.append((stats_provider_dir, resolved_models_root))
            #         # else:
            #         #     if "arcee" == str(provider_dir.name):
            #         #         print(f"NOT FOUND: {source}")  # TODO: remove

            matching_stats_provider_models_roots = []
            for stats_provider_dir, resolved_models_root in stats_provider_models_roots:
                if stats_provider_dir.name != base_model_provider:
                    continue

                source = stats_provider_dir / "models" / base_model_provider / f"{base_model_slug}.json"
                if source.is_file():
                    matching_stats_provider_models_roots.append((stats_provider_dir, resolved_models_root))
                # else:
                #     if "arcee" == str(provider_dir.name):
                #         print(f"NOT FOUND: {source}")  # TODO: remove

            for stats_provider_dir, resolved_models_root in matching_stats_provider_models_roots:
                source = stats_provider_dir / "models" / base_model_provider / f"{base_model_slug}.json"
                if not source.is_file():
                    # print(f"NOT FOUND: {source}")  # TODO: remove
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
