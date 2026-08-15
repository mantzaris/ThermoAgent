#!/usr/bin/env python3
"""Normalize only v3 text blobs and record byte/semantic provenance.

The immutable source is read from the v3 scientific snapshot. Frozen v1/v2
paths are never selected. The script is idempotent and refuses to overwrite a
v3 file whose semantics differ from the snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


V3_SNAPSHOT = "3f844966930b1cfb5a43bdf3a4d3e744391d1018"
V3_PREFIX = "results/human_operator_v3/"
TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".txt", ".csv", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".tex", ".svg", ".log",
}


def _git_bytes(root: Path, specification: str) -> bytes:
    return subprocess.run(
        ["git", "show", specification],
        cwd=str(root),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _tracked_v3_paths(root: Path) -> List[str]:
    output = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", V3_SNAPSHOT, "--", V3_PREFIX],
        cwd=str(root),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if Path(line).suffix.lower() in TEXT_SUFFIXES]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _semantic_value(path: Path, value: bytes) -> Tuple[str, Any]:
    text = value.decode("utf-8")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv_rows", list(csv.reader(io.StringIO(text, newline="")))
    if suffix == ".json":
        return "json_value", json.loads(text)
    if suffix == ".jsonl":
        return "jsonl_values", [json.loads(line) for line in text.splitlines() if line]
    return "unicode_lines", text.splitlines()


def normalize(repository_root: Path, output: Path) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for relative in _tracked_v3_paths(repository_root):
        path = repository_root / relative
        original = _git_bytes(repository_root, V3_SNAPSHOT + ":" + relative)
        crlf_count = original.count(b"\r\n")
        if not crlf_count:
            continue
        normalized = original.replace(b"\r\n", b"\n")
        if b"\r" in normalized:
            raise RuntimeError("unpaired carriage return in %s" % relative)
        method_before, semantic_before = _semantic_value(path, original)
        method_after, semantic_after = _semantic_value(path, normalized)
        semantic_equal = method_before == method_after and semantic_before == semantic_after
        if not semantic_equal:
            raise RuntimeError("normalization changed parsed semantics for %s" % relative)
        current = path.read_bytes()
        if current not in (original, normalized):
            raise RuntimeError("refusing to overwrite modified v3 artifact %s" % relative)
        path.write_bytes(normalized)
        records.append({
            "path": relative,
            "original_v3_sha256": _sha256(original),
            "normalized_v4_branch_sha256": _sha256(normalized),
            "original_crlf_sequences": crlf_count,
            "semantic_method": method_before,
            "semantic_equal": semantic_equal,
            "source_snapshot": V3_SNAPSHOT,
        })
    report = {
        "scope": "v3 text artifacts containing CRLF; v1/v2 excluded",
        "source_snapshot": V3_SNAPSHOT,
        "files_normalized": len(records),
        "all_semantically_equal": all(row["semantic_equal"] for row in records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/human_operator_v4/reproducibility/v3_line_ending_normalization.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = normalize(root, output)
    print(json.dumps({
        "files_normalized": report["files_normalized"],
        "all_semantically_equal": report["all_semantically_equal"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
