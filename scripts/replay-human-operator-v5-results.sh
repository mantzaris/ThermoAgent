#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$repo_dir/results/human_operator_v5"
mkdir -p "$results_dir/logs"
cd "$repo_dir"
python3 -c '
from pathlib import Path
from thermoagent.v5_replay import replay_v5_results
r=replay_v5_results(Path("results/human_operator_v5"))
print({k:r[k] for k in ("episodes_replayed","failures","mismatches","maximum_conservation_residual")})
raise SystemExit(1 if r["mismatches"] else 0)
' 2>&1 | tee "$results_dir/logs/replay.log"
