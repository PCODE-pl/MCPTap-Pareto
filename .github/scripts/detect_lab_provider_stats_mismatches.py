#!/usr/bin/env python3
"""Detect statistics models missing from their matching first-party provider catalog."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
STATISTICS_SOURCES = ("openrouter", "vercel", "llmgateway")


@dataclass(frozen=True)
class StatsReport:
    """Identify one valid statistics document and its provider/model identity."""

    source: str
    reported_provider: str
    model_id: str
    path: str


@dataclass(frozen=True)
class Finding:
    """Describe a lab-provider model absent from the provider catalog."""

    model_id: str
    lab_provider: str
    canonical_model_exists: bool
    lab_provider_exists: bool
    lab_provider_model_exists: bool
    reports: tuple[StatsReport, ...]


@dataclass(frozen=True)
class LabProviderIndex:
    """Index one lab provider's model paths and inherited model identities."""

    exists: bool
    exact_ids: frozenset[str]
    base_model_ids: frozenset[str]


@dataclass(frozen=True)
class ScanResult:
    """Return detector findings and input errors."""

    reported_file_count: int
    reported_model_count: int
    findings: tuple[Finding, ...]
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "reported_file_count": self.reported_file_count,
            "reported_model_count": self.reported_model_count,
            "finding_count": len(self.findings),
            "findings": [
                {
                    **asdict(finding),
                    "reports": [asdict(report) for report in finding.reports],
                }
                for finding in self.findings
            ],
            "errors": list(self.errors),
        }


def _model_id_from_stats_path(path: pathlib.Path, source_dir: pathlib.Path) -> tuple[str, str] | None:
    relative = path.relative_to(source_dir)
    parts = relative.parts
    if len(parts) < 4 or parts[1] != "models" or path.suffix != ".json":
        return None
    model_parts = parts[2:]
    if any(part in {"", ".", ".."} or "\\" in part for part in model_parts):
        return None
    model_parts = (*model_parts[:-1], path.stem)
    return parts[0], "/".join(model_parts)


def _toml_model_id(path: pathlib.Path, models_dir: pathlib.Path) -> str:
    relative = path.relative_to(models_dir)
    filename = relative.name
    if filename.endswith(".toml"):
        filename = filename[: -len(".toml")]
    return "/".join((*relative.parent.parts, filename))


def _iter_stats_reports(
    repo_root: pathlib.Path,
) -> tuple[list[StatsReport], list[str]]:
    reports: list[StatsReport] = []
    errors: list[str] = []
    for source in STATISTICS_SOURCES:
        source_dir = repo_root / "stats" / source
        if not source_dir.is_dir():
            continue
        for path in sorted(source_dir.rglob("*.json")):
            identity = _model_id_from_stats_path(path, source_dir)
            if identity is None:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"{path}: cannot read JSON: {exc}")
                continue
            if not isinstance(payload, dict):
                errors.append(f"{path}: JSON root must be an object")
                continue
            reported_provider, model_id = identity
            reports.append(
                StatsReport(
                    source=source,
                    reported_provider=reported_provider,
                    model_id=model_id,
                    path=path.relative_to(repo_root).as_posix(),
                )
            )
    return reports, errors


def _provider_model_ids(
    provider_models_dir: pathlib.Path,
) -> tuple[set[str], set[str], list[str]]:
    exact_ids: set[str] = set()
    base_model_ids: set[str] = set()
    errors: list[str] = []
    if not provider_models_dir.is_dir():
        return exact_ids, base_model_ids, errors
    for path in sorted(provider_models_dir.rglob("*.toml")):
        exact_ids.add(_toml_model_id(path, provider_models_dir))
        try:
            document = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{path}: cannot read TOML: {exc}")
            continue
        base_model = document.get("base_model")
        if isinstance(base_model, str) and base_model.strip():
            base_model_ids.add(base_model.strip())
    return exact_ids, base_model_ids, errors


def _load_lab_provider_index(
    repo_root: pathlib.Path,
    lab_provider: str,
) -> tuple[LabProviderIndex, list[str]]:
    provider_dir = repo_root / "providers" / lab_provider
    exact_ids, base_model_ids, errors = _provider_model_ids(provider_dir / "models")
    return (
        LabProviderIndex(
            exists=provider_dir.is_dir(),
            exact_ids=frozenset(exact_ids),
            base_model_ids=frozenset(base_model_ids),
        ),
        errors,
    )


def _build_finding(
    repo_root: pathlib.Path,
    model_id: str,
    reports: tuple[StatsReport, ...],
    provider_index: LabProviderIndex,
) -> Finding:
    model_parts = model_id.split("/")
    lab_provider = model_parts[0]
    canonical_model_path = repo_root / "models" / pathlib.Path(*model_parts[:-1], f"{model_parts[-1]}.toml")
    model_id_is_exact = "/".join(model_parts[1:]) in provider_index.exact_ids
    return Finding(
        model_id=model_id,
        lab_provider=lab_provider,
        canonical_model_exists=canonical_model_path.is_file(),
        lab_provider_exists=provider_index.exists,
        lab_provider_model_exists=model_id_is_exact or model_id in provider_index.base_model_ids,
        reports=reports,
    )


def scan_repository(repo_root: pathlib.Path = REPO_ROOT) -> ScanResult:
    """Scan all configured statistics sources against matching lab providers."""
    reports, errors = _iter_stats_reports(repo_root)
    reports_by_model: dict[str, list[StatsReport]] = defaultdict(list)
    for report in reports:
        reports_by_model[report.model_id].append(report)

    provider_indexes: dict[str, LabProviderIndex] = {}
    findings: list[Finding] = []
    for model_id in sorted(reports_by_model):
        lab_provider = model_id.split("/", maxsplit=1)[0]
        if lab_provider not in provider_indexes:
            provider_index, provider_errors = _load_lab_provider_index(repo_root, lab_provider)
            provider_indexes[lab_provider] = provider_index
            errors.extend(provider_errors)
        model_reports = tuple(
            sorted(
                reports_by_model[model_id],
                key=lambda report: (report.source, report.reported_provider, report.path),
            )
        )
        finding = _build_finding(repo_root, model_id, model_reports, provider_indexes[lab_provider])
        if not finding.lab_provider_model_exists:
            findings.append(finding)

    return ScanResult(
        reported_file_count=len(reports),
        reported_model_count=len(reports_by_model),
        findings=tuple(findings),
        errors=tuple(sorted(set(errors))),
    )


def _format_finding(finding: Finding) -> Iterable[str]:
    if finding.lab_provider_exists:
        status = "model missing from lab provider"
    else:
        status = "lab provider directory missing"
    canonical = "present" if finding.canonical_model_exists else "missing"
    yield f"- {finding.model_id}: {status} (lab_provider={finding.lab_provider}, canonical model: {canonical})"
    for report in finding.reports:
        yield f"    {report.source}/{report.reported_provider}: {report.path}"


def print_text(result: ScanResult) -> None:
    print("Lab provider statistics mismatch detector")
    print("==========================================")
    print(f"Valid statistics files: {result.reported_file_count}")
    print(f"Reported canonical models: {result.reported_model_count}")
    print(f"Lab-provider mismatches: {len(result.findings)}")
    if result.findings:
        print("\nFindings")
        print("--------")
        for finding in result.findings:
            print(*_format_finding(finding), sep="\n")
    if result.errors:
        print(f"\nErrors: {len(result.errors)}")
        for error in result.errors:
            print(f"  - {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete report as JSON",
    )
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Return a nonzero status when any mismatch is found",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = scan_repository()
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n")
    else:
        print_text(result)
    if result.errors:
        return 1
    return 1 if args.fail_on_findings and result.findings else 0


if __name__ == "__main__":
    sys.exit(main())
