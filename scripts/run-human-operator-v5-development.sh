#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$repo_dir/results/human_operator_v5"
mkdir -p "$results_dir/logs"
cd "$repo_dir"
python3 -c '
from pathlib import Path
from thermoagent.v5_experiments import run_matrix
root=Path(".").resolve(); results=root/"results"/"human_operator_v5"
print(run_matrix(root, results, "development_primary_v2",
    ("commercial","humanitarian","utility_restoration"),
    ("nominal","isolated_physical","telemetry_integrity","partition","correlated","compound","ood"),
    ("private_fragmented","public_shared"), tuple(range(51101,51121)),
    ("event_triggered",), resume=True))
' 2>&1 | tee "$results_dir/logs/development_reproduction.log"
