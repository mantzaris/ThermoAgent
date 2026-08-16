"""V7 artifact indexing, checksum verification, and text hygiene."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .events import sha256_file
from .v5_experiments import atomic_json, write_csv


TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".toml", ".svg"}


def _artifact_type(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "vector_figure"
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        return "preview"
    if path.name.endswith("events.jsonl.gz"):
        return "compressed_event_ledger"
    return path.suffix.lower().lstrip(".") or "file"


def build_index(results_root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    self_referential = {
        "INDEX.csv", "artifact_index_summary.json",
        "artifact_index_verification.json", "crlf_audit.json",
    }
    for path in sorted(results_root.rglob("*")):
        if (
            not path.is_file() or path.name in self_referential
            or path.name.endswith(".tmp")
        ):
            continue
        relative = path.relative_to(results_root)
        parts = relative.parts
        stage = parts[0] if len(parts) > 1 else "root"
        application = ""
        method = ""
        text = str(relative)
        for candidate in ("humanitarian", "utility_restoration"):
            if candidate in text:
                application = candidate
        for candidate in (
            "combined_generalized_entropic", "generalized_entropic",
            "strongest_nonentropic", "kpi_confidence", "always_act",
            "event_triggered", "always_on",
        ):
            if candidate in text:
                method = candidate
                break
        rows.append({
            "relative_path": str(relative),
            "artifact_type": _artifact_type(path),
            "stage": stage,
            "application": application,
            "method": method,
            "description": path.stem.replace("_", " "),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    write_csv(results_root / "INDEX.csv", rows)
    report = {
        "indexed_artifacts_excluding_index": len(rows),
        "total_size_bytes": sum(value["size_bytes"] for value in rows),
        "maximum_artifact_size_bytes": max(
            (value["size_bytes"] for value in rows), default=0,
        ),
        "artifacts_over_50_mib": sum(
            value["size_bytes"] > 50 * 1024 * 1024 for value in rows
        ),
    }
    atomic_json(results_root / "reproducibility" / "artifact_index_summary.json", report)
    return report


def verify_index(results_root: Path) -> Dict[str, Any]:
    import csv

    rows = list(csv.DictReader(
        (results_root / "INDEX.csv").open("r", encoding="utf-8", newline=""),
    ))
    missing = []
    mismatched = []
    for row in rows:
        path = results_root / row["relative_path"]
        if not path.is_file():
            missing.append(row["relative_path"])
        elif sha256_file(path) != row["sha256"]:
            mismatched.append(row["relative_path"])
    report = {
        "indexed_rows": len(rows), "missing": missing,
        "checksum_mismatches": mismatched,
        "pass": not missing and not mismatched,
    }
    atomic_json(results_root / "reproducibility" / "artifact_index_verification.json", report)
    return report


def crlf_audit(repository: Path, results_root: Path) -> Dict[str, Any]:
    roots = [
        repository / "thermoagent", repository / "scripts",
        repository / "configs", repository / "notes", results_root,
    ]
    failures = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                if b"\r\n" in path.read_bytes():
                    failures.append(str(path.relative_to(repository)))
    report = {"files_checked_roots": len(roots), "crlf_files": failures, "pass": not failures}
    atomic_json(results_root / "reproducibility" / "crlf_audit.json", report)
    return report
