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
from thermoagent.v6_analysis import (
    analyze_entropy_family_ablations, analyze_risk_dataset,
    refit_permutation_family_test, run_crossfit_dynamic_evaluation,
)
from thermoagent.v6_communication import analyze_sketch_stage
from thermoagent.v6_learnability import run_learnability_diagnostics

repo = Path.cwd()
root = repo / "results" / "generalized_entropic_consensus_v6"
candidate = root / "development" / "formal_reference" / "candidate_decisions.csv"
risk_root = root / "development"
print(analyze_risk_dataset(candidate, risk_root, "risk_analysis"))
print(analyze_entropy_family_ablations(candidate, root / "development" / "entropy_family"))
print(refit_permutation_family_test(candidate, root / "development" / "permutation", 200, 66070))
print(run_learnability_diagnostics(root))
print(run_crossfit_dynamic_evaluation(
    repo, root, candidate,
    root / "development" / "risk_analysis" / "risk_analysis.json",
))
print(analyze_sketch_stage(
    root / "development" / "sketch_reference",
    root / "development" / "communication",
))
PY
