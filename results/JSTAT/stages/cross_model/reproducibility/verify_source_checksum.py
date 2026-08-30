#!/usr/bin/env python3
"""Verify V15 semantic source and reconstruct its documented legacy checksum."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_entries(repository: Path) -> List[Tuple[str, str]]:
    roots = (
        repository / "thermoagent/statmech_llm_v15",
        repository / "configs/statmech_v15",
        repository / "tests/statmech_v15",
    )
    files = [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
        and path.name != "protocol_frozen.yaml"
    ]
    files.extend((repository / "scripts").glob("*statmech-v15*"))
    return sorted(
        (path.relative_to(repository).as_posix(), sha256_file(path))
        for path in set(files)
        if path.is_file()
    )


def manifest_digest(entries: List[Tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative_path, file_digest in sorted(entries):
        digest.update(
            relative_path.encode("utf-8")
            + b"\0"
            + file_digest.encode("ascii")
            + b"\0"
        )
    return digest.hexdigest()


def atomic_json(payload: Dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(destination.parent),
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)


def main() -> int:
    repository = Path(__file__).resolve().parents[3]
    record_path = Path(__file__).with_name("source_checksum_audit.json")
    record: Dict[str, object] = json.loads(record_path.read_text(encoding="utf-8"))
    cache = record["legacy_cache_entry"]
    assert isinstance(cache, dict)

    entries = semantic_entries(repository)
    semantic = manifest_digest(entries)
    legacy_entries = entries + [(str(cache["relative_path"]), str(cache["sha256"]))]
    legacy = manifest_digest(legacy_entries)
    protocol = sha256_file(repository / "configs/statmech_v15/protocol_frozen.yaml")

    cache_path = str(cache["relative_path"])
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", cache_path],
        cwd=str(repository),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", cache_path],
        cwd=str(repository),
        check=False,
    ).returncode == 0
    index_path = repository / "results/collective_agent_statmech_v15/INDEX.csv"
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    missing_indexed_files = [
        row["relative_path"]
        for row in index_rows
        if not (repository / row["relative_path"]).is_file()
    ]
    mismatched_indexed_files = [
        row["relative_path"]
        for row in index_rows
        if (repository / row["relative_path"]).is_file()
        and (
            sha256_file(repository / row["relative_path"]) != row["sha256"]
            or (repository / row["relative_path"]).stat().st_size
            != int(row["bytes"])
        )
    ]
    forbidden_suffixes = {
        ".jsonl",
        ".safetensors",
        ".pt",
        ".bin",
        ".npy",
        ".npz",
        ".tar",
        ".zip",
        ".png",
    }
    indexed_paths = [repository / row["relative_path"] for row in index_rows]
    checks = {
        "semantic_source_matches": semantic == record["semantic_source_sha256"],
        "legacy_source_reconstructed": legacy
        == record["legacy_execution_source_sha256"],
        "protocol_matches": protocol == record["protocol_sha256"],
        "legacy_cache_not_tracked": not tracked,
        "legacy_cache_ignored": ignored,
        "current_index_complete": not missing_indexed_files,
        "current_index_checksums_match": not mismatched_indexed_files,
        "current_index_no_oversized_files": not any(
            path.is_file() and path.stat().st_size > 10 * 1024 * 1024
            for path in indexed_paths
        ),
        "current_index_no_forbidden_artifacts": not any(
            path.suffix.lower() in forbidden_suffixes for path in indexed_paths
        ),
    }
    legacy_verification_path = record_path.with_name("verification.json")
    legacy_verification = json.loads(
        legacy_verification_path.read_text(encoding="utf-8")
    )
    legacy_checks = dict(legacy_verification.get("checks", {}))
    non_source_checks = {
        key: bool(value)
        for key, value in legacy_checks.items()
        if key != "execution_source_matches_freeze"
    }
    checks["legacy_package_non_source_checks_pass"] = bool(non_source_checks) and all(
        non_source_checks.values()
    )
    output = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "legacy_direct_execution_source_check": bool(
            legacy_checks.get("execution_source_matches_freeze", False)
        ),
        "legacy_verification_status": legacy_verification.get("status", "missing"),
        "interpretation": (
            "The clean package passes when the documented reconstruction replaces only "
            "the legacy direct check for the absent ignored cache. All other package checks "
            "are required to pass unchanged."
        ),
        "semantic_source_sha256": semantic,
        "reconstructed_legacy_execution_source_sha256": legacy,
        "protocol_sha256": protocol,
        "semantic_source_files": len(entries),
        "indexed_files": len(index_rows),
        "missing_indexed_files": missing_indexed_files,
        "mismatched_indexed_files": mismatched_indexed_files,
    }
    atomic_json(output, record_path.with_name("verification_clean.json"))
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
