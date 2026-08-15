#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python3 -c '
import csv, hashlib, json
from pathlib import Path
root = Path("results/human_operator_v5")
index_path = root / "INDEX.csv"
rows = list(csv.DictReader(index_path.open(encoding="utf-8", newline="")))
indexed = {row["path"] for row in rows}
actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != index_path}
failures = []
if indexed != actual:
    failures.append({"missing_from_index": sorted(actual-indexed), "missing_from_tree": sorted(indexed-actual)})
for row in rows:
    path = root / row["path"]
    if not path.is_file():
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if int(row["size_bytes"]) != path.stat().st_size or row["sha256"] != digest:
        failures.append({"path": row["path"], "reason": "size_or_sha256_mismatch"})
freeze = json.loads((root/"protocol"/"development_freeze_manifest.json").read_text(encoding="utf-8"))
config = Path(freeze["canonical_config"])
if hashlib.sha256(config.read_bytes()).hexdigest() != freeze["canonical_config_sha256"]:
    failures.append({"path": str(config), "reason": "frozen_protocol_checksum_mismatch"})
print({"indexed_artifacts": len(rows), "failures": len(failures), "protocol_checksum_verified": not any(f.get("reason") == "frozen_protocol_checksum_mismatch" for f in failures)})
if failures:
    print(failures[:20])
    raise SystemExit(1)
'
