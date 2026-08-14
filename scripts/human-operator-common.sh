#!/usr/bin/env bash
# Shared local/RunPod v3 command setup. Source this file; do not execute it.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"
v3_results="${THERMO_HUMAN_RESULTS:-$repo_dir/results/human_operator_v3}"
mkdir -p "$v3_results/logs"

run_human_command() {
  local log_name="$1"
  shift
  local log_path="$v3_results/logs/$log_name"
  mkdir -p "$(dirname "$log_path")"
  set +e
  "$python_bin" -m thermoagent.human_cli \
    --root "$repo_dir" --results "$v3_results" "$@" 2>&1 | tee -a "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e
  (( status == 0 )) || {
    echo "ThermoHITL command failed with status $status; retained log: $log_path" >&2
    return "$status"
  }
}
