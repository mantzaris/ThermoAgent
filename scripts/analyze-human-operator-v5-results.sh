#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$repo_dir/results/human_operator_v5"
mkdir -p "$results_dir/logs"
cd "$repo_dir"
python3 -c '
from pathlib import Path
from thermoagent.v5_analysis import analyze_development, analyze_sketch_ablation
root=Path("results/human_operator_v5")
print(analyze_development(root, stage="development_primary_v2", permutation_replicates=199))
print(analyze_sketch_ablation(root))
' 2>&1 | tee "$results_dir/logs/analysis_reproduction.log"
gzip -9 -f "$results_dir/statistics/candidate_crossfit_predictions.csv"
