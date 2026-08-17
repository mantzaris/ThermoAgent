"""Artifact indexing and integrity checks for the isolated V8 namespace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .v5_experiments import atomic_json, write_csv


TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".toml"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_v8_index(results_root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    crlf = []
    oversized = []
    for path in sorted(value for value in results_root.rglob("*") if value.is_file()):
        relative = path.relative_to(results_root)
        if relative.as_posix() in ("INDEX.csv", "reproducibility/checksums/artifact_manifest.json"):
            continue
        size = path.stat().st_size
        if size >= 50 * 1024 * 1024:
            oversized.append(str(relative))
        if path.suffix.lower() in TEXT_SUFFIXES and b"\r\n" in path.read_bytes():
            crlf.append(str(relative))
        parts = relative.parts
        stage = parts[0] if len(parts) > 1 else "root"
        rows.append({
            "relative_path": str(relative),
            "artifact_type": path.suffix.lower().lstrip(".") or "file",
            "stage": stage,
            "application": (
                "humanitarian" if "humanitarian" in path.name
                else "utility_restoration" if "utility_restoration" in path.name
                else "both_or_not_applicable"
            ),
            "method": "multiple_or_not_applicable",
            "description": "V8 reproducibility artifact",
            "size_bytes": size,
            "sha256": _sha256(path),
        })
    write_csv(results_root / "INDEX.csv", rows)
    manifest = {
        "namespace": str(results_root), "indexed_artifacts": len(rows),
        "total_indexed_bytes": int(sum(value["size_bytes"] for value in rows)),
        "crlf_text_files": crlf, "oversized_files": oversized,
        "status": "pass" if not crlf and not oversized else "fail",
        "artifacts": [
            {key: value for key, value in row.items() if key in (
                "relative_path", "size_bytes", "sha256",
            )}
            for row in rows
        ],
    }
    atomic_json(
        results_root / "reproducibility" / "checksums" / "artifact_manifest.json",
        manifest,
    )
    return manifest


def verify_v8_index(results_root: Path) -> Dict[str, Any]:
    path = results_root / "reproducibility" / "checksums" / "artifact_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mismatches = []
    for row in manifest["artifacts"]:
        artifact = results_root / row["relative_path"]
        if not artifact.exists() or artifact.stat().st_size != row["size_bytes"]:
            mismatches.append(row["relative_path"] + ":missing_or_size")
        elif _sha256(artifact) != row["sha256"]:
            mismatches.append(row["relative_path"] + ":sha256")
    return {
        "indexed_artifacts": len(manifest["artifacts"]),
        "mismatches": mismatches,
        "status": "pass" if not mismatches else "fail",
    }
