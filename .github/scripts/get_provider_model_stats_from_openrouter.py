#!/usr/bin/env python3
"""Collect OpenRouter endpoint statistics for provider-backed model TOMLs."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import tomllib

DEFAULT_API_URL = "https://openrouter.ai/api/v1/models"
# DEFAULT_AI_MODEL = "google/gemini-2.5-flash"
DEFAULT_AI_MODEL = "meta-llama/llama-3.3-70b-instruct"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE_FILE_NAME = ".openrouter_provider_mapping_cache.json"
STATS_KEYS = (
    "uptime_last_30m",
    "uptime_last_5m",
    "uptime_last_1d",
    "latency_last_30m",
    "throughput_last_30m",
)


def parse_args() -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default=os.environ.get("OPENROUTER_MODELS_URL", DEFAULT_API_URL),
        help="OpenRouter models API base URL",
    )
    parser.add_argument(
        "--provider-dir",
        type=pathlib.Path,
        default=repo_root / "providers",
        help="Root directory containing provider.toml files",
    )
    parser.add_argument(
        "--openrouter-models-dir",
        type=pathlib.Path,
        default=repo_root / "providers" / "openrouter" / "models",
        help="Directory containing OpenRouter provider model TOMLs",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=repo_root / "stats" / "openrouter",
        help="Output directory for provider/model JSON files",
    )
    parser.add_argument(
        "--routers-dir",
        type=pathlib.Path,
        default=repo_root / "routers",
        help="Directory containing router TOML files",
    )
    parser.add_argument(
        "--synthetic-output-dir",
        type=pathlib.Path,
        default=repo_root / "stats" / "_synthetic" / "openrouter",
        help="Output directory for synthetic provider/model JSON files",
    )
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
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def provider_name_variants(value: str) -> set[str]:
    normalized = normalize_text(value)
    variants = {normalized}
    for suffix in ("ai", "api", "cloud", "llc", "inc", "direct", "router"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
            variants.add(normalized[: -len(suffix)])
    return variants


def load_router_providers_from_models_dir_struct(routers_dir: pathlib.Path) -> tuple[set[str], list[str]]:
    routers: set[str] = set()
    errors: list[str] = []
    for router_file in sorted(routers_dir.glob("*.toml")):
        try:
            document = tomllib.loads(router_file.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{router_file}: {exc}")
            continue
        if router_file.stem != "openrouter" and document.get("router_providers_from_models_dir_struct") is True:
            routers.add(router_file.stem)
    return routers, errors


def router_slug_matches(left: str, right: str) -> bool:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    return bool(normalized_left) and normalized_left == normalized_right


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


def write_synthetic_stats(
    *,
    providers_dir: pathlib.Path,
    routers_dir: pathlib.Path,
    stats_dir: pathlib.Path,
    synthetic_dir: pathlib.Path,
    dry_run: bool,
) -> tuple[int, list[str], list[str]]:
    routers, errors = load_router_providers_from_models_dir_struct(routers_dir)
    collisions: list[str] = []
    written = 0
    if not stats_dir.is_dir() or not providers_dir.is_dir():
        return written, errors, collisions
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
        matching_routers = sorted(router for router in routers if router_slug_matches(router, provider_dir.name))
        if len(matching_routers) != 1:
            if len(matching_routers) > 1:
                errors.append(f"{provider_dir}: matches multiple routers: {', '.join(matching_routers)}")
            continue
        models_dir = provider_dir / "models"
        if not models_dir.is_dir():
            continue

        for model_file in sorted(models_dir.rglob("*.toml")):
            try:
                model_document = tomllib.loads(model_file.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError) as exc:
                errors.append(f"{model_file}: {exc}")
                continue
            parsed_base_model = parse_base_model(model_document.get("base_model"))
            if parsed_base_model is None:
                continue
            base_model_provider, base_model_slug = parsed_base_model
            model_relative_path = model_file.relative_to(models_dir)
            source_candidates: dict[pathlib.Path, pathlib.Path] = {}

            for component in model_relative_path.parent.parts:
                for stats_provider_dir, resolved_models_root in stats_provider_models_roots:
                    if not router_slug_matches(component, stats_provider_dir.name):
                        continue
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


def load_openrouter_models(
    models_dir: pathlib.Path,
) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    import tomllib

    by_base_model: dict[str, list[dict[str, str]]] = {}
    errors: list[str] = []
    for path in sorted(models_dir.rglob("*.toml")):
        try:
            relative = path.relative_to(models_dir)
            filename = relative.name
            model_id = "/".join((*relative.parent.parts, filename[: -len(".toml")]))
            document = tomllib.loads(path.read_text(encoding="utf-8"))
            base_model = document.get("base_model")
            if not isinstance(base_model, str) or not base_model:
                continue
            by_base_model.setdefault(base_model, []).append({"path": str(path), "model_id": model_id})
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    return by_base_model, errors


def fetch_model_endpoints(api_url: str, model_id: str) -> list[dict[str, Any]]:
    encoded_model_id = urllib.parse.quote(model_id, safe="/")
    url = f"{api_url.rstrip('/')}/{encoded_model_id}/endpoints"
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"OpenRouter endpoint request failed for {model_id}: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter endpoint request failed for {model_id}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenRouter endpoint returned invalid JSON for {model_id}") from exc

    if not isinstance(payload, dict):
        raise TypeError(f"OpenRouter endpoint response is not an object for {model_id}")
    data = payload.get("data")
    if isinstance(data, dict):
        endpoints = data.get("endpoints")
    else:
        endpoints = data
    if not isinstance(endpoints, list):
        raise TypeError(f"OpenRouter endpoint response has no endpoints list for {model_id}")
    return [endpoint for endpoint in endpoints if isinstance(endpoint, dict)]


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
    return matches[0] if len(matches) == 1 else None


def load_mapping_cache(repo_root: pathlib.Path) -> dict[str, str | None]:
    path = repo_root / CACHE_FILE_NAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(key): value if isinstance(value, str) or value is None else None for key, value in data.items()}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def save_mapping_cache(repo_root: pathlib.Path, cache: dict[str, str | None]) -> None:
    path = repo_root / CACHE_FILE_NAME
    path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def query_provider_mappings(
    provider_names: list[str],
    providers: dict[str, dict[str, str]],
    model_name: str,
    api_key: str,
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
    request = urllib.request.Request(
        OPENROUTER_CHAT_URL,
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


def select_endpoint(endpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not endpoints:
        return None
    return min(
        endpoints,
        key=lambda item: (
            str(item.get("provider_name", "")).lower(),
            str(item.get("name", "")).lower(),
            str(item.get("tag", "")).lower(),
        ),
    )


def extract_stats(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: endpoint.get(key) for key in STATS_KEYS}


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
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    providers = load_providers(args.provider_dir)
    models, parse_errors = load_openrouter_models(args.openrouter_models_dir)
    cache = {} if args.clear_cache else load_mapping_cache(repo_root)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    source_model_ids = sorted({entry["model_id"] for entries in models.values() for entry in entries})
    endpoint_cache: dict[str, list[dict[str, Any]]] = {}
    api_errors: list[str] = []
    for model_id in source_model_ids:
        try:
            endpoint_cache[model_id] = fetch_model_endpoints(args.api_url, model_id)
        except RuntimeError as exc:
            api_errors.append(str(exc))

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
    for name in provider_names:
        deterministic = resolve_provider_deterministically(name, providers)
        if deterministic is not None:
            mapping[name] = deterministic
        elif name in cache and cache[name] is not None:
            mapping[name] = cache[name]
        else:
            unresolved_names.append(name)

    if unresolved_names and not args.disable_ai and api_key:
        ai_mapping = query_provider_mappings(unresolved_names, providers, args.ai_model, api_key)
        mapping.update(ai_mapping)
        cache.update(ai_mapping)
    elif unresolved_names and not args.disable_ai:
        print(
            "Warning: OPENROUTER_API_KEY is not set; unresolved provider names will be skipped.",
            file=sys.stderr,
        )

    if not args.dry_run and cache:
        save_mapping_cache(repo_root, cache)

    written = 0
    unmatched_providers: set[str] = set()
    multiple_endpoints: list[str] = []
    output_collisions: list[str] = []
    outputs: dict[str, dict[str, Any]] = {}

    for canonical_model, entries in sorted(models.items()):
        # Prefer a source model ID identical to the canonical model ID when available.
        source_entry = next(
            (entry for entry in entries if entry["model_id"] == canonical_model),
            min(entries, key=lambda entry: entry["model_id"]),
        )
        source_model_id = source_entry["model_id"]
        for endpoint in endpoint_cache.get(source_model_id, []):
            provider_name = endpoint.get("provider_name")
            if not isinstance(provider_name, str):
                continue
            canonical_provider = mapping.get(provider_name)
            if canonical_provider is None:
                unmatched_providers.add(provider_name)
                continue
            matched_endpoints = [
                item for item in endpoint_cache.get(source_model_id, []) if item.get("provider_name") == provider_name
            ]
            if len(matched_endpoints) > 1:
                multiple_endpoints.append(
                    f"{provider_name}/{canonical_model}: {len(matched_endpoints)} endpoints; selected deterministic first"
                )
            output_key = f"{canonical_provider}/models/{canonical_model}.json"
            if output_key in outputs:
                output_collisions.append(output_key)
                continue
            outputs[output_key] = extract_stats(select_endpoint(matched_endpoints) or endpoint)

    if not args.dry_run:
        for relative_path, stats in outputs.items():
            destination = args.output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written += 1

    synthetic_written, synthetic_errors, synthetic_collisions = write_synthetic_stats(
        providers_dir=args.provider_dir,
        routers_dir=args.routers_dir,
        stats_dir=args.output_dir,
        synthetic_dir=args.synthetic_output_dir,
        dry_run=args.dry_run,
    )

    print("OpenRouter provider model stats")
    print("================================")
    print(f"OpenRouter TOMLs with base_model: {sum(len(v) for v in models.values())}")
    print(f"Unique source model IDs: {len(source_model_ids)}")
    print(f"Provider definitions: {len(providers)}")
    print(f"Provider names from endpoints: {len(provider_names)}")
    print(f"Resolved output files: {len(outputs)}")
    print(f"Written: {written}" if not args.dry_run else f"Would write: {len(outputs)}")
    print(
        f"Synthetic written: {synthetic_written}" if not args.dry_run else f"Synthetic would write: {synthetic_written}"
    )
    print_bucket("Unmatched provider names", sorted(set(unmatched_providers)))
    print_bucket("Multiple endpoints for provider/model", sorted(set(multiple_endpoints)))
    print_bucket("Output collisions", sorted(set(output_collisions)))
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
