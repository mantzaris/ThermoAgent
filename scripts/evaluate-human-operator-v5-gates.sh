#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python3 -c '
from pathlib import Path
from thermoagent.v5_gates import evaluate_v5_gates
root=Path("results/human_operator_v5")
print(evaluate_v5_gates(root, root/"reproducibility"/"test_summary.json"))
'
