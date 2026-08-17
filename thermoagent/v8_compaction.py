"""Losslessly pack completed V8 raw stages with per-member provenance."""

from __future__ import annotations

import gzip
import concurrent.futures
import hashlib
import io
import json
import lzma
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .events import sha256_file
from .v5_experiments import atomic_json, write_csv


def _pack_task(
    task: tuple[Path, Path, str, Sequence[Path]],
) -> Dict[str, Any]:
    return _pack_application(*task)


def _logical_bytes(path: Path) -> tuple[bytes, str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read(), "gzip"
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as handle:
            return handle.read(), "xz"
    return path.read_bytes(), "plain"


def _logical_name(relative: Path) -> str:
    value = str(relative)
    if value.endswith(".gz"):
        return value[:-3]
    if value.endswith(".xz"):
        return value[:-3]
    return value


def _application_for_run(path: Path) -> str:
    name = path.name
    for application in ("humanitarian", "utility_restoration"):
        if "-%s-" % application in name:
            return application
    raise ValueError("cannot infer V8 application from run directory: %s" % path)


def _pack_application(
    stage_dir: Path,
    destination: Path,
    application: str,
    run_dirs: Sequence[Path],
) -> Dict[str, Any]:
    files = [path for run_dir in run_dirs for path in sorted(run_dir.rglob("*")) if path.is_file()]
    if not files:
        raise RuntimeError("no %s files found for packed stage" % application)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + ".tmp")
    rows: List[Dict[str, Any]] = []
    # Preset 6 captures the large cross-run redundancy while keeping stage
    # packing operationally tractable; preset 9 was profiled and abandoned
    # before any source removal because it was disproportionately slow.
    with tarfile.open(temporary, mode="w:xz", preset=6) as archive:
        for path in files:
            relative = path.relative_to(stage_dir)
            payload, source_compression = _logical_bytes(path)
            logical_name = _logical_name(relative)
            info = tarfile.TarInfo(name=logical_name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
            rows.append({
                "application": application,
                "original_relative_path": str(relative),
                "archive_member": logical_name,
                "source_compression": source_compression,
                "stored_sha256": sha256_file(path),
                "logical_sha256": hashlib.sha256(payload).hexdigest(),
                "logical_size_bytes": len(payload),
            })
    temporary.replace(destination)
    observed = {}
    with tarfile.open(destination, mode="r|xz") as archive:
        for member in archive:
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError("packed V8 member is not a regular file")
            payload = handle.read()
            observed[member.name] = (
                hashlib.sha256(payload).hexdigest(), len(payload)
            )
    failures = [
        row["archive_member"]
        for row in rows
        if observed.get(row["archive_member"])
        != (row["logical_sha256"], row["logical_size_bytes"])
    ]
    if failures or len(observed) != len(rows):
        raise RuntimeError("packed V8 round-trip verification failed: %s" % failures[:5])
    manifest_path = destination.with_suffix(".manifest.csv")
    write_csv(manifest_path, rows)
    return {
        "application": application,
        "archive": str(destination),
        "archive_sha256": sha256_file(destination),
        "archive_size_bytes": destination.stat().st_size,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "members": len(rows),
        "runs": len(run_dirs),
        "logical_bytes": int(sum(row["logical_size_bytes"] for row in rows)),
        "roundtrip_verified": True,
    }


def pack_completed_stage(results_root: Path, stage: str) -> Dict[str, Any]:
    """Pack a completed raw stage, then remove only verified unpacked members."""
    stage_dir = (results_root / "raw" / stage).resolve()
    expected_parent = (results_root / "raw").resolve()
    if stage_dir.parent != expected_parent or not stage_dir.is_dir():
        raise ValueError("V8 stage packing target is not an exact raw-stage directory")
    execution = results_root / stage / "execution_summary.json"
    if not execution.exists():
        raise RuntimeError("stage cannot be packed before its execution summary exists")
    status = json.loads(execution.read_text(encoding="utf-8"))
    if int(status.get("completed_episodes", 0)) <= 0:
        raise RuntimeError("stage completion count is invalid")
    packed_dir = results_root / "raw" / "packed" / stage
    tasks = []
    # Large humanitarian ledgers can exceed 50 MiB even in a 24-run pilot
    # archive. Twelve-run chunks keep every Git-facing archive below the hard
    # ceiling for pilot and formal stages alike; archive boundaries are
    # scientifically immaterial because every member has an integrity row.
    chunk_size = 12
    for application in ("humanitarian", "utility_restoration"):
        application_runs = [
            path for path in sorted(stage_dir.iterdir())
            if path.is_dir() and _application_for_run(path) == application
        ]
        for chunk_index, start in enumerate(range(0, len(application_runs), chunk_size), 1):
            run_chunk = application_runs[start : start + chunk_size]
            tasks.append((
                stage_dir,
                packed_dir / (
                    "%s-part%02d.tar.xz" % (application, chunk_index)
                ),
                application,
                run_chunk,
            ))
    if len(tasks) >= 4:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            reports = list(executor.map(_pack_task, tasks, chunksize=1))
    else:
        reports = [_pack_task(task) for task in tasks]
    if sum(int(value["runs"]) for value in reports) != int(status["completed_episodes"]):
        raise RuntimeError("packed V8 run count does not match stage completion")
    if any(int(value["archive_size_bytes"]) >= 50 * 1024 * 1024 for value in reports):
        raise RuntimeError("packed V8 archive exceeds the 50 MiB Git-facing limit")
    # Deletion is narrowly scoped to the exact verified stage. Every original
    # logical byte is recoverable from a checksum-verified archive member.
    for path in sorted(stage_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    stage_dir.rmdir()
    report = {
        "stage": stage,
        "status": "pass",
        "archives": reports,
        "episodes": int(status["completed_episodes"]),
        "all_roundtrips_verified": all(value["roundtrip_verified"] for value in reports),
        "maximum_archive_size_bytes": max(value["archive_size_bytes"] for value in reports),
        "unpacked_stage_removed_after_verification": True,
        "recovery": "extract the application tar.xz beneath raw/%s" % stage,
    }
    atomic_json(results_root / "reproducibility" / "compaction" / (stage + ".json"), report)
    return report


def pack_retained_incomplete_stage(
    results_root: Path, stage: str, *, expected_complete: int,
    expected_partial: int,
) -> Dict[str, Any]:
    """Losslessly retain an invalidated pre-freeze stage with partial runs."""
    stage_dir = (results_root / "raw" / stage).resolve()
    expected_parent = (results_root / "raw").resolve()
    if stage_dir.parent != expected_parent or not stage_dir.is_dir():
        raise ValueError("retained V8 packing target is not an exact raw-stage directory")
    run_dirs = [value for value in sorted(stage_dir.iterdir()) if value.is_dir()]
    complete = sum((value / "episode.json").exists() for value in run_dirs)
    partial = len(run_dirs) - complete
    if complete != int(expected_complete) or partial != int(expected_partial):
        raise RuntimeError(
            "retained run accounting changed: complete=%d partial=%d" % (complete, partial)
        )
    packed_dir = results_root / "raw" / "packed" / stage
    tasks = []
    # The invalidated suppression run contains a few unusually large
    # humanitarian ledgers.  The earlier pre-hysteresis stage is safe at eight
    # runs per archive, whereas the suppression stage requires six to keep the
    # *observed* archive size below the Git-facing ceiling.  This choice changes
    # only archive boundaries; member bytes and checksums remain identical.
    chunk_size = (
        6 if stage == "development_final_hysteresis_suppression_invalidated"
        else 8
    )
    for application in ("humanitarian", "utility_restoration"):
        application_runs = [
            value for value in run_dirs if _application_for_run(value) == application
        ]
        # Invalidated pre-repair ledgers retain a wider full-ledger scope than
        # later delta ledgers. Archive limits are enforced again after packing,
        # before any unpacked source is removed.
        for chunk_index, start in enumerate(
            range(0, len(application_runs), chunk_size), 1,
        ):
            tasks.append((
                stage_dir,
                packed_dir / ("%s-part%02d.tar.xz" % (application, chunk_index)),
                application, application_runs[start : start + chunk_size],
            ))
    if len(tasks) >= 4:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            reports = list(executor.map(_pack_task, tasks, chunksize=1))
    else:
        reports = [_pack_task(task) for task in tasks]
    if sum(int(value["runs"]) for value in reports) != len(run_dirs):
        raise RuntimeError("retained V8 run count does not match packed reports")
    if any(int(value["archive_size_bytes"]) >= 50 * 1024 * 1024 for value in reports):
        raise RuntimeError("retained V8 archive exceeds the 50 MiB Git-facing limit")
    for path in sorted(stage_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    stage_dir.rmdir()
    report = {
        "stage": stage, "status": "pass", "archives": reports,
        "complete_episodes": complete, "partial_run_directories": partial,
        "all_roundtrips_verified": all(value["roundtrip_verified"] for value in reports),
        "maximum_archive_size_bytes": max(value["archive_size_bytes"] for value in reports),
        "unpacked_stage_removed_after_verification": True,
        "eligible_as_scientific_evidence": False,
    }
    atomic_json(
        results_root / "reproducibility" / "compaction" / (stage + ".json"), report,
    )
    return report
