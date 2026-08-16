#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p "$repo_dir/results/generalized_entropic_consensus_v6/logs"
exec 9>"${TMPDIR:-/tmp}/thermoagent-v6-writer.lock"
flock -n 9 || { echo "Another V6 writer is active; refusing concurrent execution." >&2; exit 75; }
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"

"$python_bin" - <<'PY'
from pathlib import Path
from thermoagent.v6_experiments import run_matrix

repo = Path.cwd()
root = repo / "results" / "generalized_entropic_consensus_v6"
print(run_matrix(
    repo, root, "development_formal_reference",
    ("commercial", "humanitarian", "utility_restoration"),
    ("nominal", "isolated_physical", "telemetry_integrity", "partition", "correlated", "compound", "ood"),
    ("private_fragmented", "public_shared"), tuple(range(66101, 66131)),
    ("never_act",), (0.5,), ("event_triggered",), 0,
))
print(run_matrix(
    repo, root, "development_sketch_reference",
    ("commercial", "humanitarian", "utility_restoration"),
    ("isolated_physical", "telemetry_integrity", "partition", "correlated", "compound", "ood"),
    ("private_fragmented",), tuple(range(66101, 66111)),
    ("never_act",), (0.5,), ("none", "periodic", "event_triggered", "always_on"), 0,
))
PY
