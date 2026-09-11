#!/usr/bin/env python3
"""Live-test free provider models and update the "free" branch of tested_models.json.

For every (lab_model, provider, provider_model) triple in the freshly
compiled pareto.json whose cost is fully zero, probe the provider API
with a tiny "jaki model?" request: first the OpenAI Responses endpoint,
then (only if that fails) chat/completions. A triple is tested when the
first successful probe returns HTTP 200 with non-empty text; its
wall-time latency in milliseconds and the winning endpoint type
("responses" or "chat/completions") are recorded. Other top-level
branches of the output file are preserved untouched; triples that do
not answer simply do not land in the "free" branch.

The output file also carries a top-level "providers" key: the sorted
list of provider names whose API key is present in the bulk
PROVIDERS_API_KEYS secret (recomputed on every run).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
PARETO_PATH = REPO_ROOT / "pareto.json"
OUTPUT_PATH = REPO_ROOT / "tested_models.json"

REQUEST_PROMPT = "jaki model?"
REQUEST_TIMEOUT_S = 30
MAX_COMPLETION_TOKENS = 16
FREE_BRANCH = "free"
BULK_KEYS_ENV = "PROVIDERS_API_KEYS"

# Providers whose provider.toml lacks the api field; endpoints come from
# the provider's documented OpenAI-compatible base URL.
FALLBACK_API_URLS = {
    "aihubmix": "https://aihubmix.com/v1",
    "vercel": "https://ai-gateway.vercel.sh/v1",
}


def load_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"Failed to parse {path}: {exc}", file=sys.stderr)
        return {}


def provider_api_base(repo_root: Path, provider: str) -> str:
    data = load_toml(repo_root / "providers" / provider / "provider.toml")
    api = str(data.get("api") or "").strip().rstrip("/")
    if api:
        return api
    return FALLBACK_API_URLS.get(provider, "")


def provider_env_var(repo_root: Path, provider: str) -> str:
    data = load_toml(repo_root / "providers" / provider / "provider.toml")
    env_vars = data.get("env")
    if isinstance(env_vars, list) and env_vars and isinstance(env_vars[0], str):
        return env_vars[0]
    return ""


def keyed_providers(repo_root: Path, api_keys: dict[str, str]) -> list[str]:
    """Return sorted provider names whose api key is present in the bulk secret."""
    providers_dir = repo_root / "providers"
    names = []
    if providers_dir.is_dir():
        for provider_dir in sorted(providers_dir.iterdir()):
            if not provider_dir.is_dir():
                continue
            if provider_env_var(repo_root, provider_dir.name) in api_keys:
                names.append(provider_dir.name)
    return names


def collect_free_triples(pareto_data: dict) -> list[tuple[str, str, str]]:
    """Return sorted (lab_model, provider, provider_model) triples with fully zero cost."""
    triples: set[tuple[str, str, str]] = set()
    for lab_model, entry in pareto_data.get("stats", {}).items():
        if not isinstance(entry, dict):
            continue
        for provider, models in entry.get("providers", {}).items():
            if not isinstance(models, dict):
                continue
            for provider_model, info in models.items():
                cost = info.get("cost", {}) if isinstance(info, dict) else {}
                if isinstance(cost, dict) and cost.get("input") == 0 and cost.get("output") == 0:
                    triples.add((lab_model, provider, provider_model))
    return sorted(triples)


def post_json(url: str, api_key: str, payload: dict, timeout_s: float = REQUEST_TIMEOUT_S) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def extract_text(payload: dict) -> str:
    """Extract non-empty text from either a chat/completions or responses body."""
    for choice in payload.get("choices") or []:
        text = (choice.get("message") or {}).get("content")
        if isinstance(text, str) and text.strip():
            return text
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str) and content["text"].strip():
                return content["text"]
    text = payload.get("output_text")
    if isinstance(text, str) and text.strip():
        return text
    return ""


def test_triple(api_base: str, api_key: str, provider_model: str) -> tuple[int, str] | None:
    """Probe one provider model; return (latency_ms, endpoint_type) or None on failure.

    Probe order: /responses first; chat/completions runs only when the
    responses probe did not succeed.
    """
    probes = [
        ("responses", f"{api_base}/responses", {"model": provider_model, "input": REQUEST_PROMPT}),
        (
            "chat/completions",
            f"{api_base}/chat/completions",
            {"model": provider_model, "messages": [{"role": "user", "content": REQUEST_PROMPT}], "max_tokens": 16},
        ),
    ]
    for endpoint_type, url, payload in probes:
        started = time.monotonic()
        try:
            status, body = post_json(url, api_key, payload)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  probe {url} failed: {exc}", file=sys.stderr)
            continue
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if status != 200:
            print(f"  probe {url} -> HTTP {status}", file=sys.stderr)
            continue
        try:
            text = extract_text(json.loads(body))
        except json.JSONDecodeError:
            print(f"  probe {url} returned non-JSON body", file=sys.stderr)
            continue
        if text.strip():
            return elapsed_ms, endpoint_type
        print(f"  probe {url} returned 200 with empty text", file=sys.stderr)
    return None


def resolve_api_keys(env: dict[str, str]) -> dict[str, str]:
    """Parse the bulk PROVIDERS_API_KEYS secret into an env_var -> api_key map.

    There is no fallback to individual *_API_KEY environment variables: the
    bulk secret is the only key source. A missing, invalid, or non-object
    payload is a fatal error — without it no provider can be tested.
    """
    raw = env.get(BULK_KEYS_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{BULK_KEYS_ENV} is not set; bulk secret is required to test free models")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{BULK_KEYS_ENV} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{BULK_KEYS_ENV} is not a JSON object")
    return {str(name): str(value).strip() for name, value in data.items() if str(value).strip()}


def test_free_models(
    repo_root: Path,
    pareto_data: dict,
    existing: dict | None = None,
    env: dict[str, str] | None = None,
    tester=None,
) -> dict:
    """Rebuild only the "free" branch; preserve every other top-level branch of existing."""
    env = dict(os.environ) if env is None else env
    api_keys = resolve_api_keys(env)
    preserved = {k: v for k, v in (existing or {}).items() if k != FREE_BRANCH}
    free_section: dict = {}
    for lab_model, provider, provider_model in collect_free_triples(pareto_data):
        api_base = provider_api_base(repo_root, provider)
        if not api_base:
            print(f"skip {provider}/{provider_model}: no API endpoint", file=sys.stderr)
            continue
        env_var = provider_env_var(repo_root, provider)
        api_key = api_keys.get(env_var, "")
        if not api_key:
            print(f"skip {provider} {provider_model}: missing {env_var} in {BULK_KEYS_ENV}", file=sys.stderr)
            continue
        outcome = (
            test_triple(api_base, api_key, provider_model)
            if tester is None
            else tester(api_base, api_key, provider_model)
        )
        if outcome is None:
            print(f"not tested {provider} {provider_model} (lab: {lab_model}): both probes failed", file=sys.stderr)
            continue
        latency_ms, endpoint_type = outcome
        print(f"tested {provider} {provider_model} -> {latency_ms} ms ({endpoint_type})")
        free_section.setdefault(lab_model, {"providers": {}})["providers"].setdefault(provider, {})[provider_model] = {
            "latency_ms": latency_ms,
            "endpoint_type": endpoint_type,
        }
    return {
        **preserved,
        FREE_BRANCH: {
            lab: {"providers": dict(sorted(info["providers"].items()))} for lab, info in sorted(free_section.items())
        },
    }


def load_pareto(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_output(path: Path) -> dict:
    """Load the current tested_models.json; a missing or invalid file yields an empty payload."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Reading {path} failed ({exc}); starting from an empty payload", file=sys.stderr)
        return {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    pareto_data = load_pareto(PARETO_PATH)
    existing = load_output(OUTPUT_PATH)
    result = test_free_models(REPO_ROOT, pareto_data, existing=existing)
    result["providers"] = keyed_providers(REPO_ROOT, resolve_api_keys(dict(os.environ)))
    free_section = result[FREE_BRANCH]
    tested_count = sum(len(models) for info in free_section.values() for models in info["providers"].values())
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preserved_count = len(result) - 1
    print(f"Wrote {OUTPUT_PATH} ({tested_count} tested free model triples; {preserved_count} other branches preserved)")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
