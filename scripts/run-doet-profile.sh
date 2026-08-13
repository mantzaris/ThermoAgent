#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export THERMO_CACHE="${THERMO_CACHE:-/workspace/.cache/thermoagent}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/thermoagent-pycache}"

"$python_bin" -m thermoagent model-smoke \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --output results/entropy_triggered_v2/logs/setup/model_smoke.json
"$python_bin" -m thermoagent sweep \
  --config configs/entropy_trigger_profile.yaml \
  --results results/entropy_triggered_v2 \
  --root "$repo_dir"
"$python_bin" -m thermoagent replay \
  --results results/entropy_triggered_v2 --stages profile_v2 \
  --report-name profile_v2_replay_report.json
