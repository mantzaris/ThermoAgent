"""Deterministic inventories for the publication-oriented repository."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd


FINAL_SCRIPT_ROLES: Mapping[str, str] = {
    "setup-study-environment.sh": "install and verify the pinned GPU environment",
    "prefetch-models.py": "download and verify pinned model revisions outside Git",
    "run-formal-experiment.sh": "deliberately execute one frozen model panel family",
    "replay-results.sh": "replay retained external decision records",
    "analyze-results.sh": "rebuild aggregate analyses and effective-model comparisons",
    "generate-figures.sh": "regenerate the fourteen publication figures",
    "verify-results.sh": "run the retained test and scientific-integrity suite",
    "run-tests.sh": "run the consolidated Python tests",
    "build-jstat-paper.sh": "build the canonical JSTAT manuscript",
    "verify-jstat-paper-assets.sh": "verify scientific hashes and PDF assets",
    "compare-reconstruction.py": "compare reconstructed and retained aggregate science",
    "verify-source-checksum.py": "verify frozen and consolidated source provenance",
}

TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".yaml",
    ".yml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_records(paths: Iterable[Path], repository: Path) -> List[tuple[str, str]]:
    records: List[tuple[str, str]] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        records.append((path.relative_to(repository).as_posix(), text))
    return records


def _active_reference_records(repository: Path) -> List[tuple[str, str]]:
    roots = (
        repository / "README.md",
        repository / "docs",
        repository / "pyproject.toml",
        repository / "thermoagent/statmech_llm",
        repository / "tests/statmech_llm",
        repository / "configs/statmech_llm",
        repository / "paper/JSTAT",
    )
    paths: List[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    for name in FINAL_SCRIPT_ROLES:
        path = repository / "scripts" / name
        if path.is_file():
            paths.append(path)
    return _text_records(paths, repository)


def _historical_reference_records(repository: Path) -> List[tuple[str, str]]:
    roots = (
        repository / "results/JSTAT/stages",
        repository / "results/JSTAT/reproducibility",
    )
    paths = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]
    return _text_records(paths, repository)


def _script_category(name: str) -> str:
    lowered = name.lower()
    if "human-operator" in lowered:
        return "abandoned human-operator workflow"
    if "doet" in lowered:
        return "abandoned logistics workflow"
    if any(token in lowered for token in ("v6", "v7", "v8", "entropy-trigger")):
        return "superseded entropy-controller workflow"
    if any("statmech-v%d" % value in lowered for value in range(9, 12)):
        return "superseded early statistical-mechanics workflow"
    if any("statmech-v%d" % value in lowered for value in range(12, 16)):
        return "consolidated final-study stage wrapper"
    if "runpod" in lowered:
        return "retired remote orchestration helper"
    if any(token in lowered for token in ("holdout", "pilot", "sweep", "calibration")):
        return "abandoned pre-publication experiment workflow"
    return "abandoned legacy repository workflow"


def _script_replacement(name: str) -> str:
    lowered = name.lower()
    if "setup" in lowered:
        return "scripts/setup-study-environment.sh"
    if "prefetch" in lowered:
        return "scripts/prefetch-models.py"
    if "reconstruction" in lowered and "analysis" in lowered:
        return "scripts/replay-results.sh; scripts/analyze-results.sh; scripts/compare-reconstruction.py"
    if "reconstruction" in lowered and "formal" in lowered:
        return "scripts/run-formal-experiment.sh; scripts/verify-source-checksum.py"
    if "replay" in lowered:
        return "scripts/replay-results.sh"
    if "analy" in lowered or "surrogate" in lowered:
        return "scripts/analyze-results.sh"
    if "figure" in lowered or "render" in lowered:
        return "scripts/generate-figures.sh"
    if "paper" in lowered:
        return "scripts/build-jstat-paper.sh"
    if "test" in lowered:
        return "scripts/run-tests.sh"
    if "verify" in lowered or "validate" in lowered or "pdf" in lowered:
        return "scripts/verify-results.sh"
    if "formal" in lowered:
        return "scripts/run-formal-experiment.sh"
    if "build" in lowered or "report" in lowered:
        return "scripts/analyze-results.sh; scripts/verify-results.sh"
    if "freeze" in lowered:
        return "configs/statmech_llm/*/protocol.yaml (already frozen)"
    return "Git history; no current publication workflow role"


def script_inventory(repository: Path) -> pd.DataFrame:
    """Classify every current top-level script from a positive dependency closure."""

    repository = Path(repository).resolve()
    active_records = _active_reference_records(repository)
    historical_records = _historical_reference_records(repository)
    rows: List[Dict[str, object]] = []
    for path in sorted((repository / "scripts").iterdir()):
        if not path.is_file():
            continue
        relative = path.relative_to(repository).as_posix()
        needles = (relative, path.name)
        active_references = sorted(
            name
            for name, text in active_records
            if name != relative and any(needle in text for needle in needles)
        )
        historical_references = sorted(
            name
            for name, text in historical_records
            if any(needle in text for needle in needles)
        )
        retain = path.name in FINAL_SCRIPT_ROLES
        category = "final publication workflow" if retain else _script_category(path.name)
        replacement = relative if retain else _script_replacement(path.name)
        rows.append(
            {
                "current_path": relative,
                "proposed_final_path": relative if retain else "",
                "disposition": "retain" if retain else "delete_after_reference_check",
                "referenced_by": "; ".join(active_references),
                "historical_record_references": "; ".join(historical_references),
                "scientific_role": FINAL_SCRIPT_ROLES.get(path.name, category),
                "evidence_deletion_safe": (
                    "retained in positive final workflow closure"
                    if retain
                    else (
                        "no active reference; functionality replaced by %s; historical manifests are records, not runtime dependencies"
                        % replacement
                        if not active_references
                        else "NOT YET SAFE: active reference remains"
                    )
                ),
                "replacement": replacement,
                "sha256": _sha256(path),
                "bytes": int(path.stat().st_size),
            }
        )
    return pd.DataFrame(rows)


def figure_inventory(repository: Path) -> pd.DataFrame:
    """Hash and classify all retained semantic-stage figure artifacts."""

    repository = Path(repository).resolve()
    canonical = sorted((repository / "paper/JSTAT/figures").glob("*.pdf"))
    canonical_by_hash = {_sha256(path): path for path in canonical}
    stage_files = sorted(
        path
        for path in (repository / "results/JSTAT/stages").glob("*/figures/**/*")
        if path.is_file()
    )
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in stage_files:
        groups[_sha256(path)].append(path)
    reference_records = _text_records(
        [
            path
            for root in (
                repository / "results/JSTAT",
                repository / "paper/JSTAT",
                repository / "README.md",
                repository / "docs",
                repository / "thermoagent/statmech_llm",
                repository / "tests/statmech_llm",
            )
            for path in (
                [root]
                if root.is_file()
                else list(root.rglob("*")) if root.exists() else []
            )
            if path.is_file()
        ],
        repository,
    )
    rows: List[Dict[str, object]] = []
    for path in stage_files:
        relative = path.relative_to(repository).as_posix()
        digest = _sha256(path)
        group = groups[digest]
        canonical_match = canonical_by_hash.get(digest)
        retained_copy = ""
        if canonical_match is not None:
            classification = "1 canonical-paper byte-identical duplicate"
            retained_copy = canonical_match.relative_to(repository).as_posix()
        elif len(group) > 1 and path != group[0]:
            classification = "2 retained-result byte-identical duplicate"
            retained_copy = group[0].relative_to(repository).as_posix()
        elif any(
            token in path.name.lower()
            for token in ("preview", "render_page", "font_report", "intermediate")
        ) or path.suffix.lower() == ".png":
            classification = "4 temporary rendering artifact"
        elif path.suffix.lower() in {".pdf", ".csv"}:
            classification = "3 unique scientific or provenance artifact"
        else:
            classification = "5 unclear"
        references = sorted(
            name
            for name, text in reference_records
            if name != relative and (relative in text or path.name in text)
        )
        rows.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": digest,
                "classification": classification,
                "identical_retained_copy": retained_copy,
                "referenced_by": "; ".join(references),
                "disposition": (
                    "retain_manifest_or_qa_copy"
                    if classification.startswith("1 ")
                    else "retain_by_unique_or_unclear_policy"
                    if classification.startswith(("3 ", "5 "))
                    else "retain_documented_duplicate"
                    if classification.startswith("2 ")
                    else "review_before_deletion"
                ),
            }
        )
    return pd.DataFrame(rows)


def write_repository_inventories(repository: Path) -> Dict[str, object]:
    """Write the pre-deletion scientific-figure and script audit records."""

    repository = Path(repository).resolve()
    destination = repository / "results/JSTAT/provenance"
    destination.mkdir(parents=True, exist_ok=True)
    figures = figure_inventory(repository)
    scripts = script_inventory(repository)
    figures.to_csv(destination / "figure_inventory.csv", index=False, lineterminator="\n")
    scripts.to_csv(destination / "script_inventory.csv", index=False, lineterminator="\n")
    summary = {
        "inventory_version": "publication-consolidation-1.0",
        "figure_artifacts": int(len(figures)),
        "figure_classifications": {
            str(key): int(value)
            for key, value in figures["classification"].value_counts().items()
        },
        "canonical_paper_figures": 14,
        "stage_figures_deleted": 0,
        "scripts_examined": int(len(scripts)),
        "scripts_retained": int((scripts["disposition"] == "retain").sum()),
        "scripts_safe_to_delete_after_reference_check": int(
            (scripts["disposition"] == "delete_after_reference_check").sum()
        ),
        "unsafe_script_candidates": int(
            scripts["evidence_deletion_safe"].str.startswith("NOT YET SAFE").sum()
        ),
    }
    (destination / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "FINAL_SCRIPT_ROLES",
    "figure_inventory",
    "script_inventory",
    "write_repository_inventories",
]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(write_repository_inventories(root), indent=2, sort_keys=True))
