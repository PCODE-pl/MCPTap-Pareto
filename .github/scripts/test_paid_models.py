#!/usr/bin/env python3
"""Live-probe paid provider models and update the "paid" branch of tested_models.json.

For every (lab_model, provider, provider_model) triple in the freshly
compiled pareto.json whose cost is not fully zero, but only for
providers whose API key is present in the bulk PROVIDERS_API_KEYS
secret, send minimal one-character probes: POST /responses with
"input": ".", then POST /chat/completions with a single "user" message
".". The probe generates no completion, so the cost stays zero.

Whether a probe answer proves the model exists is provider-specific
(live-verified 2026-09-11) and configured in PROVIDER_DETERMINANTS:

- "402/403" (default): HTTP 402 (no-credit anti-abuse guard) or 403
  (insufficient balance / model busy) proves the model exists — both
  statuses are only returned after the provider validated the model
  name. Used for zenmux, orcarouter, aihubmix, unorouter.
- "models": the provider's credit gate answers 402 even for
  nonexistent models, so probes cannot distinguish; the authoritative
  GET /models catalogue is used instead (kilo).
- Excluded providers (EXCLUDED_PROVIDERS) are skipped entirely:
  opencode and opencode-go sit behind a Cloudflare gate that answers
  403 to everything, vercel answers 403 card-gate before validating
  the model, and nvidia actually generates (billed) output for an
  existing model. Their triples never land in the "paid" branch.

Each probe runs with a short timeout; probes for all triples run in a
small thread pool. Other top-level branches of the output file are
preserved untouched; triples never proven to exist simply do not land
in the "paid" branch.

The output file also carries a top-level "providers" key: the sorted
list of provider names whose API key is present in the bulk
PROVIDERS_API_KEYS secret (recomputed on every run).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
PARETO_PATH = REPO_ROOT / "pareto.json"
OUTPUT_PATH = REPO_ROOT / "tested_models.json"

REQUEST_TIMEOUT_S = 12
MAX_WORKERS = 8
PAID_BRANCH = "paid"
BULK_KEYS_ENV = "PROVIDERS_API_KEYS"

# Determinant strategy per provider. Anything not listed here uses the
# default "402/403" probe rule. "models" resolves existence from the
# provider's GET /models catalogue instead of probing inference.
DEFAULT_DETERMINANT = "402/403"
PROVIDER_DETERMINANTS: dict[str, str] = {
    "kilo": "models",
}

# Providers that cannot be tested truthfully with zero-cost probes and
# are therefore skipped entirely (their triples never enter "paid").
EXCLUDED_PROVIDERS: dict[str, str] = {
    "opencode": "Cloudflare gate answers 403 to every request",
    "opencode-go": "same Cloudflare gate as opencode",
    "vercel": "card-gate 403 fires before model validation",
    "nvidia": "an existing model actually generates (billed) output for the probe",
}

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


def collect_paid_triples(pareto_data: dict) -> list[tuple[str, str, str]]:
    """Return sorted (lab_model, provider, provider_model) triples with non-zero cost.

    Entries without a usable cost dict (including malformed non-dict
    values) are skipped: an absent cost cannot be proven non-zero.
    """
    triples: set[tuple[str, str, str]] = set()
    for lab_model, entry in pareto_data.get("stats", {}).items():
        if not isinstance(entry, dict):
            continue
        for provider, models in entry.get("providers", {}).items():
            if not isinstance(models, dict):
                continue
            for provider_model, info in models.items():
                cost = info.get("cost", {}) if isinstance(info, dict) else {}
                if not isinstance(cost, dict):
                    continue
                input_cost = cost.get("input")
                output_cost = cost.get("output")
                # A triple counts as paid only when a cost value is
                # explicitly present and non-zero; absent values (or an
                # empty cost dict) are not provably paid and are skipped.
                if (input_cost is not None and input_cost != 0) or (output_cost is not None and output_cost != 0):
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


def test_triple(
    api_base: str,
    api_key: str,
    provider_model: str,
    exists_codes: tuple[int, ...] = (402, 403),
) -> str | None:
    """Probe one provider model with a one-character minimal request.

    Probe order: /responses first; chat/completions runs only when the
    responses probe did not prove the model exists. A status from
    exists_codes (402 no-credit guard, 403 insufficient balance /
    model busy — returned only after model-name validation) proves the
    model exists; every other answer proves it does not. Returns the
    winning endpoint type or None.
    """
    probes = [
        ("responses", f"{api_base}/responses", {"model": provider_model, "input": "."}),
        (
            "chat/completions",
            f"{api_base}/chat/completions",
            {"model": provider_model, "messages": [{"role": "user", "content": "."}]},
        ),
    ]
    for endpoint_type, url, payload in probes:
        try:
            status, _body = post_json(url, api_key, payload)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  probe {url} failed: {exc}", file=sys.stderr)
            continue
        if status in exists_codes:
            return endpoint_type
        print(f"  probe {url} -> HTTP {status} (treated as model missing)", file=sys.stderr)
    return None


def fetch_model_ids(api_base: str, api_key: str) -> set[str] | None:
    """Fetch the provider's GET /models catalogue; None on failure."""
    request = urllib.request.Request(f"{api_base}/models", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        print(f"  GET {api_base}/models failed: {exc}", file=sys.stderr)
        return None
    ids = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(ids, list):
        print(f"  GET {api_base}/models returned unexpected payload", file=sys.stderr)
        return None
    model_ids = {m.get("id") for m in ids if isinstance(m, dict) and isinstance(m.get("id"), str)}
    return {str(mid) for mid in model_ids if mid is not None}


def resolve_api_keys(env: dict[str, str]) -> dict[str, str]:
    """Parse the bulk PROVIDERS_API_KEYS secret into an env_var -> api_key map.

    There is no fallback to individual *_API_KEY environment variables: the
    bulk secret is the only key source. A missing, invalid, or non-object
    payload is a fatal error — without it no provider can be tested.
    """
    raw = env.get(BULK_KEYS_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{BULK_KEYS_ENV} is not set; bulk secret is required to test paid models")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{BULK_KEYS_ENV} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{BULK_KEYS_ENV} is not a JSON object")
    return {str(name): str(value).strip() for name, value in data.items() if str(value).strip()}


def _probe_one(
    repo_root: Path,
    lab_model: str,
    provider: str,
    provider_model: str,
    api_keys: dict[str, str],
    tester=None,
) -> tuple[str, str, str, str | None]:
    """Probe a single triple; returns (lab, provider, model, endpoint_type-or-None)."""
    if provider in EXCLUDED_PROVIDERS:
        print(f"skip {provider} {provider_model}: provider excluded ({EXCLUDED_PROVIDERS[provider]})", file=sys.stderr)
        return lab_model, provider, provider_model, None
    api_base = provider_api_base(repo_root, provider)
    if not api_base:
        print(f"skip {provider}/{provider_model}: no API endpoint", file=sys.stderr)
        return lab_model, provider, provider_model, None
    env_var = provider_env_var(repo_root, provider)
    api_key = api_keys.get(env_var, "")
    if not api_key:
        print(f"skip {provider} {provider_model}: missing {env_var} in {BULK_KEYS_ENV}", file=sys.stderr)
        return lab_model, provider, provider_model, None
    if tester is not None:
        outcome = tester(api_base, api_key, provider_model)
    elif PROVIDER_DETERMINANTS.get(provider, DEFAULT_DETERMINANT) == "models":
        model_ids = fetch_model_ids(api_base, api_key)
        if model_ids is None or provider_model not in model_ids:
            print(f"  {provider} {provider_model}: not in /models catalogue", file=sys.stderr)
            outcome = None
        else:
            outcome = "models"
    else:
        outcome = test_triple(api_base, api_key, provider_model)
    return lab_model, provider, provider_model, outcome


def test_paid_models(
    repo_root: Path,
    pareto_data: dict,
    existing: dict | None = None,
    env: dict[str, str] | None = None,
    tester=None,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """Rebuild only the "paid" branch; preserve every other top-level branch of existing."""
    env = dict(os.environ) if env is None else env
    api_keys = resolve_api_keys(env)
    preserved = {k: v for k, v in (existing or {}).items() if k != PAID_BRANCH}
    paid_section: dict = {}
    triples = collect_paid_triples(pareto_data)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_probe_one, repo_root, lab_model, provider, provider_model, api_keys, tester)
            for lab_model, provider, provider_model in triples
        ]
        for future in as_completed(futures):
            lab_model, provider, provider_model, outcome = future.result()
            if outcome is None:
                print(
                    f"not tested {provider} {provider_model} (lab: {lab_model}): model not proven to exist",
                    file=sys.stderr,
                )
                continue
            endpoint_type = outcome
            print(f"tested {provider} {provider_model} ({endpoint_type})")
            paid_section.setdefault(lab_model, {"providers": {}})["providers"].setdefault(provider, {})[
                provider_model
            ] = {
                "endpoint_type": endpoint_type,
            }
    return {
        **preserved,
        PAID_BRANCH: {
            lab: {"providers": dict(sorted(info["providers"].items()))} for lab, info in sorted(paid_section.items())
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
    result = test_paid_models(REPO_ROOT, pareto_data, existing=existing)
    result["providers"] = keyed_providers(REPO_ROOT, resolve_api_keys(dict(os.environ)))
    paid_section = result[PAID_BRANCH]
    tested_count = sum(len(models) for info in paid_section.values() for models in info["providers"].values())
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preserved_count = len(result) - 1
    print(f"Wrote {OUTPUT_PATH} ({tested_count} tested paid model triples; {preserved_count} other branches preserved)")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
