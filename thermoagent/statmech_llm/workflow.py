"""Idempotent external-artifact and provenance utilities for the final study."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Sequence

import pandas as pd
import yaml


PARENT_COMMIT = "103e4c4598ecc26a98c37a8d03ee3663f9be1070"
SOURCE_SCRIPTS = (
    "setup-study-environment.sh",
    "prefetch-models.py",
    "run-formal-experiment.sh",
    "replay-results.sh",
    "analyze-results.sh",
    "generate-figures.sh",
    "verify-results.sh",
    "run-tests.sh",
    "build-jstat-paper.sh",
    "verify-jstat-paper-assets.sh",
    "compare-reconstruction.py",
    "verify-source-checksum.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifact_root() -> Path:
    root = Path(
        os.environ.get(
            "THERMOAGENT_ARTIFACT_ROOT", "/workspace/ThermoAgent-JSTAT-artifacts"
        )
    ).resolve()
    repository = repository_root()
    if root == repository or repository in root.parents:
        raise ValueError("raw study artifacts must remain outside the repository")
    return root


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )


def atomic_bytes(value: bytes, destination: Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_json(value: object, destination: Path) -> None:
    atomic_bytes(
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8"),
        destination,
    )


def atomic_csv(value: object, destination: Path) -> None:
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    try:
        rendered = frame.to_csv(index=False, lineterminator="\n")
    except TypeError:  # pandas < 1.5 used the underscored spelling.
        rendered = frame.to_csv(index=False, line_terminator="\n")
    atomic_bytes(rendered.encode("utf-8"), destination)


def load_yaml(path: Path) -> Dict[str, object]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a YAML mapping: %s" % path)
    return value


def source_files(repository: Path) -> Sequence[Path]:
    repository = Path(repository).resolve()
    roots = (
        repository / "thermoagent/statmech_llm",
        repository / "configs/statmech_llm",
        repository / "tests/statmech_llm",
    )
    files = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() not in {".pyc", ".pyo"}
        and path.name != "protocol.yaml"
    ]
    files.extend(
        repository / "scripts" / filename
        for filename in SOURCE_SCRIPTS
        if (repository / "scripts" / filename).is_file()
    )
    return tuple(sorted(set(files)))


def execution_source_checksum(repository: Path) -> str:
    repository = Path(repository).resolve()
    digest = hashlib.sha256()
    for path in source_files(repository):
        relative = path.relative_to(repository).as_posix()
        digest.update(relative.encode("utf-8") + b"\0" + sha256_file(path).encode("ascii") + b"\0")
    return digest.hexdigest()


def verify_consolidated_source(repository: Path) -> Dict[str, object]:
    """Verify the publication tree against the frozen-source bridge record.

    The formal protocol retains the checksum of the source tree that generated
    the original trajectories.  Publication consolidation changes paths and
    workflow wrappers, so a second checksum identifies the reviewable semantic
    tree without rewriting the frozen protocol.
    """

    repository = Path(repository).resolve()
    record_path = repository / "results/JSTAT/reproducibility/source_consolidation.json"
    if not record_path.is_file():
        raise RuntimeError("source-consolidation record is missing")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    protocol_path = repository / "configs/statmech_llm/cross_model/protocol.yaml"
    protocol = load_yaml(protocol_path)
    observed = execution_source_checksum(repository)
    checks = {
        "consolidated_source_matches": observed
        == str(record.get("consolidated_source_sha256", "")),
        "protocol_file_matches": sha256_file(protocol_path)
        == str(record.get("protocol_sha256", "")),
        "frozen_execution_source_preserved": str(
            protocol["provenance"]["execution_source_sha256"]  # type: ignore[index]
        )
        == str(record.get("frozen_execution_source_sha256", "")),
    }
    if not all(checks.values()):
        raise RuntimeError("consolidated execution source failed provenance checks")
    return {
        "status": "passed",
        "checks": checks,
        "consolidated_source_sha256": observed,
        "frozen_execution_source_sha256": str(
            protocol["provenance"]["execution_source_sha256"]  # type: ignore[index]
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "source_file_count": len(source_files(repository)),
        "record_path": record_path.relative_to(repository).as_posix(),
    }


def tree_digest(root: Path) -> Dict[str, object]:
    root = Path(root)
    digest = hashlib.sha256()
    count = 0
    size = 0
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8") + b"\0" + sha256_file(path).encode("ascii") + b"\0")
            count += 1
            size += int(path.stat().st_size)
    return {"file_count": count, "bytes": size, "tree_sha256": digest.hexdigest()}


@contextlib.contextmanager
def stage_lock(name: str) -> Iterator[None]:
    lock = artifact_root() / "locks" / (str(name) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("exclusive study stage lock exists: %s" % lock) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("pid=%d\nstarted=%s\n" % (os.getpid(), utc_now()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def ensure_external_layout() -> None:
    for relative in (
        "pilot",
        "formal/panels",
        "raw/pilot",
        "raw/formal/qwen",
        "raw/formal/granite",
        "invalidated",
        "analysis",
        "logs",
        "locks",
        "reproducibility",
        "pdf_qa/rendered_300dpi",
    ):
        (artifact_root() / relative).mkdir(parents=True, exist_ok=True)


__all__ = [
    "PARENT_COMMIT",
    "SOURCE_SCRIPTS",
    "artifact_root",
    "atomic_bytes",
    "atomic_csv",
    "atomic_json",
    "ensure_external_layout",
    "execution_source_checksum",
    "load_yaml",
    "repository_root",
    "sha256_file",
    "sha256_json",
    "stage_lock",
    "tree_digest",
    "utc_now",
    "verify_consolidated_source",
]
