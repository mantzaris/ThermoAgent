"""Small reproducibility helpers shared by the V11 commands."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_root() -> Path:
    value = os.environ.get("THERMO_V11_ARTIFACT_ROOT", "/workspace/ThermoAgent-v11-artifacts")
    path = Path(value).resolve()
    repository = Path(__file__).resolve().parents[2]
    if path == repository or repository in path.parents:
        raise ValueError("V11 raw artifacts must be external to the repository")
    return path


def load_yaml(path: Path) -> Dict[str, object]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_checksum(repository: Path) -> str:
    roots = [
        Path(repository) / "thermoagent/statmech_llm_v11",
        Path(repository) / "configs/statmech_v11",
        Path(repository) / "scripts",
        Path(repository) / "tests/statmech_v11",
    ]
    digest = hashlib.sha256()
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in paths:
            if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
                continue
            normalized = path.as_posix().lower()
            if "v11" not in path.name.lower() and "statmech_llm_v11" not in normalized and "statmech_v11" not in normalized:
                continue
            relative = path.relative_to(repository).as_posix()
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _atomic_bytes(payload: bytes, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(destination))
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_json(value: Mapping[str, object], destination: Path) -> None:
    _atomic_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"), destination)


def atomic_csv(rows: Sequence[Mapping[str, object]], destination: Path) -> None:
    if not rows:
        raise ValueError("refusing to create an empty evidence table")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "rows.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        _atomic_bytes(path.read_bytes(), destination)


@contextlib.contextmanager
def stage_lock(name: str) -> Iterator[None]:
    root = artifact_root() / "locks"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (str(name) + ".lock")
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("stage lock already exists: %s" % path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("pid=%d\nstarted=%s\n" % (os.getpid(), utc_now()))
        yield
    finally:
        path.unlink(missing_ok=True)


def external_checksums(root: Path) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        output.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return output
