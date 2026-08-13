#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"
freeze_path="results/entropy_triggered_v2/protocol/holdout_freeze.json"
[[ -f "$freeze_path" ]] || {
  echo "Locked holdout requires scripts/freeze-doet-holdout.sh first." >&2
  exit 2
}
"$python_bin" -m thermoagent verify-protocol \
  --root . --freeze "$freeze_path" >/dev/null
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export THERMO_CACHE="${THERMO_CACHE:-/workspace/.cache/thermoagent}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/thermoagent-pycache}"

# The sweep emits only completion counts. Do not inspect episode outcomes until
# every checksum-frozen design row finishes. Existing completed/failed
# manifests are retained and resumed without selective reruns.
"$python_bin" -m thermoagent sweep \
  --config configs/entropy_trigger_holdout_locked.yaml \
  --results results/entropy_triggered_v2 --root "$repo_dir"
"$repo_dir/scripts/replay-doet-results.sh"
"$repo_dir/scripts/analyze-doet-results.sh"
"$repo_dir/scripts/generate-doet-figures.sh"
"$python_bin" -m thermoagent index --results results/entropy_triggered_v2
