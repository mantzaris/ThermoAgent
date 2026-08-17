"""Deterministic compact artifact I/O for Git-facing V8 runs."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import lzma
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .events import sha256_file
from .events import EventLedger


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def write_csv_gzip(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Atomically write deterministic UTF-8 CSV inside a deterministic gzip."""
    destination = path if path.suffix == ".gz" else Path(str(path) + ".gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    text = io.StringIO(newline="")
    if fieldnames:
        writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fieldnames})
    payload = text.getvalue().encode("utf-8")
    temporary = Path(str(destination) + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0,
        ) as compressed:
            compressed.write(payload)
    temporary.replace(destination)
    with gzip.open(destination, "rb") as handle:
        if handle.read() != payload:
            raise RuntimeError("deterministic CSV gzip round trip failed")
    return {
        "path": str(destination),
        "rows": len(rows),
        "columns": len(fieldnames),
        "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def write_json_gzip(path: Path, value: Any) -> Dict[str, Any]:
    destination = path if path.suffix == ".gz" else Path(str(path) + ".gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = Path(str(destination) + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0,
        ) as compressed:
            compressed.write(payload)
    temporary.replace(destination)
    return {
        "path": str(destination),
        "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def read_csv_gzip(path: Path) -> List[Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_event_ledger_xz(path: Path, ledger: EventLedger) -> str:
    """Write a deterministic lossless XZ JSONL ledger and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    # Per-run ledgers are packed again across panels after stage completion.
    # Preset 0 is lossless and keeps atomic per-run persistence from dominating
    # simulation time. Completed stages are subsequently decompressed,
    # checksum-verified, and packed across panels at preset 6, so the Git-facing
    # size does not depend on this transient per-run compression level. Logical
    # JSONL bytes, row counts, and replay digests are unchanged.
    with lzma.open(temporary, "wt", encoding="utf-8", newline="", preset=0) as handle:
        for event in ledger.events:
            handle.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
    temporary.replace(path)
    # Decode every completed file before it becomes an indexed artifact.
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        decoded_rows = sum(1 for _ in handle)
    if decoded_rows != len(ledger.events):
        raise RuntimeError("XZ event-ledger round trip row count failed")
    return sha256_file(path)
