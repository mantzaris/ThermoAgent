"""External-artifact, locking, checksum, and protocol helpers for V12."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifact_root() -> Path:
    path = Path(os.environ.get("THERMO_V12_ARTIFACT_ROOT", "/workspace/ThermoAgent-v12-artifacts")).resolve()
    repository = repository_root()
    if path == repository or repository in path.parents:
        raise ValueError("V12 raw artifacts must remain outside the repository")
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


def execution_source_checksum(repository: Path) -> str:
    repository = Path(repository).resolve()
    roots = [
        repository / "thermoagent/statmech_llm_v12",
        repository / "configs/statmech_v12",
        repository / "tests/statmech_v12",
    ]
    digest = hashlib.sha256()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo") or path.name == "protocol_frozen.yaml":
                continue
            relative = path.relative_to(repository).as_posix()
            digest.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _atomic_bytes(payload: bytes, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(destination))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value: Mapping[str, object], destination: Path) -> None:
    _atomic_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"), destination)


def atomic_csv(rows: Sequence[Mapping[str, object]], destination: Path) -> None:
    if not rows:
        raise ValueError("refusing to write an empty V12 table")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fieldnames:
                fieldnames.append(str(key))
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "rows.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        _atomic_bytes(path.read_bytes(), destination)


@contextlib.contextmanager
def stage_lock(stage: str) -> Iterator[None]:
    directory = artifact_root() / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (str(stage) + ".lock")
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError("exclusive V12 stage lock already exists: %s" % path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("pid=%d\nstarted=%s\n" % (os.getpid(), utc_now()))
        yield
    finally:
        path.unlink(missing_ok=True)


def external_manifest(root: Path) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        output.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return output


def freeze_protocol(repository: Path, pilot_summary: Mapping[str, object]) -> Dict[str, object]:
    repository = Path(repository).resolve()
    template_path = repository / "configs/statmech_v12/protocol_template.yaml"
    frozen_path = repository / "configs/statmech_v12/protocol_frozen.yaml"
    if frozen_path.exists():
        frozen = load_yaml(frozen_path)
        return {
            "protocol_path": str(frozen_path),
            "protocol_sha256": sha256_file(frozen_path),
            "execution_source_sha256": frozen["provenance"]["execution_source_sha256"],  # type: ignore[index]
        }
    template = load_yaml(template_path)
    template["status"] = "frozen_before_formal_outcomes"
    template["frozen_at_utc"] = utc_now()
    template["pilot_estimability"] = dict(pilot_summary)
    template.setdefault("provenance", {})
    template["provenance"]["execution_source_sha256"] = execution_source_checksum(repository)  # type: ignore[index]
    serialized = yaml.safe_dump(template, sort_keys=False, allow_unicode=True).encode("utf-8")
    _atomic_bytes(serialized, frozen_path)
    return {
        "protocol_path": str(frozen_path),
        "protocol_sha256": sha256_file(frozen_path),
        "execution_source_sha256": template["provenance"]["execution_source_sha256"],  # type: ignore[index]
    }
