#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$repo_dir/results/human_operator_v5"
mkdir -p "$results_dir/logs"
cd "$repo_dir"
python3 -c '
from pathlib import Path
from thermoagent.v5_figures import generate
print("\n".join(generate(Path("results/human_operator_v5"))))
' 2>&1 | tee "$results_dir/logs/figures.log"
