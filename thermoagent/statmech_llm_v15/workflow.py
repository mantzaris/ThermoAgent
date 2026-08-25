"""Idempotent external-artifact and provenance utilities for V15."""

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifact_root() -> Path:
    root = Path(
        os.environ.get("THERMO_V15_ARTIFACT_ROOT", "/workspace/ThermoAgent-v15-artifacts")
    ).resolve()
    repository = repository_root()
    if root == repository or repository in root.parents:
        raise ValueError("V15 raw artifacts must remain outside the repository")
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
        repository / "thermoagent/statmech_llm_v15",
        repository / "configs/statmech_v15",
        repository / "tests/statmech_v15",
    )
    files = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() not in {".pyc", ".pyo"}
        and path.name != "protocol_frozen.yaml"
    ]
    files.extend(sorted((repository / "scripts").glob("*statmech-v15*")))
    return tuple(sorted(set(files)))


def execution_source_checksum(repository: Path) -> str:
    repository = Path(repository).resolve()
    digest = hashlib.sha256()
    for path in source_files(repository):
        relative = path.relative_to(repository).as_posix()
        digest.update(relative.encode("utf-8") + b"\0" + sha256_file(path).encode("ascii") + b"\0")
    return digest.hexdigest()


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
        raise RuntimeError("exclusive V15 stage lock exists: %s" % lock) from error
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
]
