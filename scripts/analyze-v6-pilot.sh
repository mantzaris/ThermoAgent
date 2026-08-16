#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"

stage_name="${1:-pilot_v5_reference}"
analysis_name="${2:-pilot_v5_analysis}"

"$python_bin" - "$stage_name" "$analysis_name" <<'PY'
from pathlib import Path
import sys
from thermoagent.v6_analysis import analyze_risk_dataset

stage_name, analysis_name = sys.argv[1:3]
root = Path("results/generalized_entropic_consensus_v6")
analyze_risk_dataset(
    root / "pilots" / stage_name / "candidate_decisions.csv",
    root / "pilots", analysis_name,
)
PY
