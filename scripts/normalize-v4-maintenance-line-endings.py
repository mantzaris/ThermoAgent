#!/usr/bin/env python3
"""Normalize the two reviewed V1-era CSVs on the V4 maintenance branch.

The source bytes are always read from the immutable V4 result snapshot.  The
script refuses semantic changes and records both hashes so the maintenance
commit cannot be mistaken for a scientific-data revision.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path


V4_SNAPSHOT = "8ccd27df248940fc0cbb55c43a30949de3370533"
FILES = (
    "results/smoke/episodes.csv",
    "results/smoke/history/episodes-5fe54b2403ca.csv",
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rows(value: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(value.decode("utf-8"), newline="")))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    records = []
    for relative in FILES:
        original = subprocess.run(
            ["git", "show", V4_SNAPSHOT + ":" + relative],
            cwd=str(root), check=True, stdout=subprocess.PIPE,
        ).stdout
        normalized = original.replace(b"\r\n", b"\n")
        if b"\r" in normalized or rows(original) != rows(normalized):
            raise RuntimeError("line-ending normalization changed CSV semantics")
        path = root / relative
        if path.read_bytes() not in (original, normalized):
            raise RuntimeError("refusing to overwrite modified artifact: " + relative)
        path.write_bytes(normalized)
        records.append({
            "path": relative,
            "source_snapshot": V4_SNAPSHOT,
            "original_sha256": digest(original),
            "normalized_sha256": digest(normalized),
            "original_crlf_sequences": original.count(b"\r\n"),
            "row_count_including_header": len(rows(original)),
            "semantic_equal": True,
        })
    output = root / "results/human_operator_v4/reproducibility/v4_maintenance_line_endings.json"
    output.write_text(json.dumps({
        "source_snapshot": V4_SNAPSHOT,
        "scope": "two reviewed top-level smoke CSVs; values and row order unchanged",
        "all_semantically_equal": True,
        "records": records,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
