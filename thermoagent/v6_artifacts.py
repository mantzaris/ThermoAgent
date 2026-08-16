"""V6 artifact index, checksums, size guard, and secret-name audit."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .events import sha256_file
from .v5_experiments import atomic_json, write_csv


EXCLUDED_INDEX_FILES = {
    "INDEX.csv", "reproducibility/artifact_checksums.csv",
    "reproducibility/artifact_verification.json",
}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".toml"}
FORBIDDEN_NAMES = {".env", "id_rsa", "id_ed25519", "credentials.json"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}", re.I),
)


def _stage(relative: Path) -> str:
    parts = relative.parts
    if not parts:
        return "root"
    if parts[0] in ("raw", "figures", "statistics", "tables", "training", "qwen", "development", "pilots", "protocol", "manifests", "reproducibility", "negative_results", "v5_reanalysis", "dashboard_exports", "ablations"):
        return parts[0]
    return "root"


def _description(relative: Path) -> str:
    name = relative.name.replace("_", " ")
    return "%s artifact for the V6 %s stage" % (name, _stage(relative))


def build_index(results_root: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(results_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(results_root)
        if str(relative) in EXCLUDED_INDEX_FILES or path.name.endswith(".tmp"):
            continue
        files.append(path)
    rows: List[Dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(results_root)
        rows.append({
            "path": str(relative),
            "artifact_type": path.suffix.lstrip(".") or "file",
            "stage": _stage(relative),
            "description": _description(relative),
            "application": "all" if not any(value in str(relative) for value in ("commercial", "humanitarian", "utility_restoration")) else next(value for value in ("commercial", "humanitarian", "utility_restoration") if value in str(relative)),
            "method": "multiple_or_not_applicable",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    write_csv(results_root / "INDEX.csv", rows)
    write_csv(results_root / "reproducibility" / "artifact_checksums.csv", rows)
    return {"artifacts": len(rows), "total_bytes": sum(value["size_bytes"] for value in rows)}


def verify_artifacts(results_root: Path) -> Dict[str, Any]:
    checksum_path = results_root / "reproducibility" / "artifact_checksums.csv"
    import csv
    with checksum_path.open("r", encoding="utf-8", newline="") as handle:
        expected = list(csv.DictReader(handle))
    failures: List[Dict[str, str]] = []
    large_files: List[Dict[str, Any]] = []
    secret_findings: List[Dict[str, str]] = []
    for row in expected:
        path = results_root / row["path"]
        if not path.exists():
            failures.append({"path": row["path"], "reason": "missing"})
            continue
        if sha256_file(path) != row["sha256"]:
            failures.append({"path": row["path"], "reason": "checksum_mismatch"})
        if path.stat().st_size > 50 * 1024 * 1024:
            large_files.append({"path": row["path"], "size_bytes": path.stat().st_size})
        if path.name in FORBIDDEN_NAMES or any(part in (".venv", ".cache") for part in path.parts):
            secret_findings.append({"path": row["path"], "reason": "forbidden_name"})
        if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 10 * 1024 * 1024:
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            if b"\r\n" in raw:
                failures.append({"path": row["path"], "reason": "crlf_text"})
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                secret_findings.append({"path": row["path"], "reason": "secret_pattern"})
    report = {
        "artifacts_checked": len(expected),
        "failures": len(failures),
        "failure_details": failures,
        "large_files_over_50_mib": large_files,
        "secret_findings": secret_findings,
        "passed": not failures and not large_files and not secret_findings,
    }
    atomic_json(results_root / "reproducibility" / "artifact_verification.json", report)
    return report
