"""Losslessly compact completed V7 artifacts while retaining full provenance."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .events import sha256_file
from .v5_experiments import atomic_json, write_csv
from .v7_io import episode_artifacts, read_json_artifact


def _gzip_exact(path: Path) -> Dict[str, Any]:
    destination = Path(str(path) + ".gz")
    temporary = Path(str(destination) + ".tmp")
    source_bytes = path.read_bytes()
    if not destination.exists():
        with temporary.open("wb") as raw_handle:
            # mtime=0 makes repeated compaction byte-for-byte deterministic.
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, compresslevel=9, mtime=0,
            ) as handle:
                handle.write(source_bytes)
        temporary.replace(destination)
    with gzip.open(destination, "rb") as handle:
        restored = handle.read()
    if restored != source_bytes:
        raise RuntimeError("gzip round-trip mismatch for %s" % path)
    report = {
        "plain_path": str(path),
        "compressed_path": str(destination),
        "plain_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "compressed_sha256": sha256_file(destination),
        "plain_size_bytes": len(source_bytes),
        "compressed_size_bytes": destination.stat().st_size,
    }
    return report


def _equal_value(expected: Any, observed: str) -> bool:
    if expected is None:
        return observed == ""
    if isinstance(expected, bool):
        return observed.strip().lower() == str(expected).lower()
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            parsed = float(observed)
        except ValueError:
            return False
        if not math.isfinite(float(expected)):
            return parsed == float(expected)
        return math.isclose(parsed, float(expected), rel_tol=1e-12, abs_tol=1e-12)
    return observed == str(expected)


def _verify_duplicate_csv(path: Path, candidates: List[Mapping[str, Any]]) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(candidates):
        raise RuntimeError("candidate row-count mismatch for %s" % path)
    for index, (expected, observed) in enumerate(zip(candidates, rows)):
        if set(observed) != set(expected):
            raise RuntimeError("candidate columns mismatch for %s row %d" % (path, index))
        for key, value in expected.items():
            if not _equal_value(value, observed[key]):
                raise RuntimeError(
                    "candidate semantic mismatch for %s row %d field %s"
                    % (path, index, key)
                )
    return {
        "duplicate_path": str(path),
        "duplicate_sha256": sha256_file(path),
        "duplicate_size_bytes": path.stat().st_size,
        "candidate_rows": len(rows),
        "semantic_equality": True,
    }


def compact_v7(results_root: Path) -> Dict[str, Any]:
    destination = results_root / "reproducibility" / "checksums"
    manifest_path = destination / "git_facing_compaction_manifest.csv"
    summary_path = destination / "git_facing_compaction_summary.json"
    if manifest_path.exists():
        # A manifest is written only after every gzip round trip and semantic
        # duplicate check succeeds. Finish any deletions interrupted after
        # that commit point, then perform the full verification again.
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            prior_rows = list(csv.DictReader(handle))
        for row in prior_rows:
            Path(row["plain_path"]).unlink(missing_ok=True)
            if row["duplicate_path"]:
                Path(row["duplicate_path"]).unlink(missing_ok=True)
        return verify_compaction(results_root)

    rows: List[Dict[str, Any]] = []
    before_bytes = 0
    removed_duplicate_bytes = 0
    for episode_path in sorted((results_root / "raw").glob("**/episode.json")):
        value = read_json_artifact(episode_path)
        duplicate = episode_path.parent / "candidate_decisions.csv"
        duplicate_report: Dict[str, Any] = {
            "duplicate_path": "", "duplicate_sha256": "",
            "duplicate_size_bytes": 0, "candidate_rows": len(value.get("candidates", [])),
            "semantic_equality": True,
        }
        if duplicate.exists():
            duplicate_report = _verify_duplicate_csv(duplicate, value.get("candidates", []))
            removed_duplicate_bytes += int(duplicate_report["duplicate_size_bytes"])
        compacted = _gzip_exact(episode_path)
        before_bytes += int(compacted["plain_size_bytes"])
        rows.append({"artifact_role": "episode_json", **compacted, **duplicate_report})

    aggregate_rows = []
    for path in sorted(results_root.glob("*/candidate_decisions.csv")):
        compacted = _gzip_exact(path)
        before_bytes += int(compacted["plain_size_bytes"])
        aggregate_rows.append({
            "artifact_role": "aggregate_candidate_table", **compacted,
            "duplicate_path": "", "duplicate_sha256": "",
            "duplicate_size_bytes": 0, "candidate_rows": "",
            "semantic_equality": True,
        })
    rows.extend(aggregate_rows)
    if not rows:
        raise RuntimeError("no uncompressed V7 artifacts were found and no prior manifest exists")
    write_csv(manifest_path, rows)
    # Deletions happen only after every compressed artifact, duplicate check,
    # and the provenance manifest have been written successfully.
    for row in rows:
        Path(row["plain_path"]).unlink()
        if row["duplicate_path"]:
            Path(row["duplicate_path"]).unlink()
    report = {
        "status": "pass",
        "episode_artifacts_compacted": sum(row["artifact_role"] == "episode_json" for row in rows),
        "aggregate_candidate_tables_compacted": sum(row["artifact_role"] == "aggregate_candidate_table" for row in rows),
        "duplicate_candidate_tables_removed_after_semantic_check": sum(bool(row["duplicate_path"]) for row in rows),
        "all_duplicate_tables_semantically_equal": all(bool(row["semantic_equality"]) for row in rows),
        "uncompressed_bytes_replaced": before_bytes,
        "verified_duplicate_bytes_removed": removed_duplicate_bytes,
        "manifest": str(manifest_path.relative_to(results_root)),
    }
    atomic_json(summary_path, report)
    return verify_compaction(results_root)


def verify_compaction(results_root: Path) -> Dict[str, Any]:
    manifest_path = (
        results_root / "reproducibility" / "checksums"
        / "git_facing_compaction_manifest.csv"
    )
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    failures: List[str] = []
    for row in rows:
        compressed = Path(row["compressed_path"])
        if not compressed.is_absolute():
            # Paths are recorded repository-relative by the current workflow.
            compressed = Path.cwd() / compressed
        if not compressed.exists():
            failures.append("missing:%s" % row["compressed_path"])
            continue
        if sha256_file(compressed) != row["compressed_sha256"]:
            failures.append("checksum:%s" % row["compressed_path"])
            continue
        with gzip.open(compressed, "rb") as handle:
            restored_sha = hashlib.sha256(handle.read()).hexdigest()
        if restored_sha != row["plain_sha256"]:
            failures.append("roundtrip:%s" % row["compressed_path"])
    episodes = episode_artifacts(results_root / "raw")
    report = {
        "status": "pass" if not failures else "fail",
        "manifest_rows": len(rows),
        "canonical_episode_artifacts": len(episodes),
        "uncompressed_episode_artifacts": len(list((results_root / "raw").glob("**/episode.json"))),
        "remaining_per_run_candidate_duplicates": len(list((results_root / "raw").glob("**/candidate_decisions.csv"))),
        "failures": failures,
    }
    atomic_json(
        results_root / "reproducibility" / "checksums"
        / "git_facing_compaction_verification.json",
        report,
    )
    if failures:
        raise RuntimeError("V7 compaction verification failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path,
        default=Path("results/complexity_entropic_coordination_v7"),
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    report = verify_compaction(args.results) if args.verify_only else compact_v7(args.results)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
