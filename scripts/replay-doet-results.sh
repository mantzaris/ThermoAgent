#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"
stages=(validation holdout_locked)
[[ -f results/entropy_triggered_v2/ablations/episodes.csv ]] \
  && stages+=(ablations)
exec "$python_bin" -m thermoagent replay \
  --results results/entropy_triggered_v2 \
  --stages "${stages[@]}" --report-name replay_report.json
