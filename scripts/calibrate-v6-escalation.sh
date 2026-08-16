#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"

"$python_bin" - <<'PY'
from pathlib import Path
from thermoagent.v6_analysis import calibrate_escalation_threshold

root = Path("results/generalized_entropic_consensus_v6/pilots")
calibrate_escalation_threshold(
    root / "pilot_v11_timing_final" / "candidate_decisions.csv",
    root / "pilot_v11_timing_final_analysis" / "crossfit_risk_predictions.csv.gz",
    root / "pilot_v11_timing_final_analysis",
)
PY

echo "V6 escalation threshold calibrated from pilot timing only"
