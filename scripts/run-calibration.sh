#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
"$python_bin" -m thermoagent calibrate \
  --output results/reproducibility/macrostate_calibration.json \
  --seeds 101,102,103,104,105 --horizon 24
"$python_bin" -m thermoagent select-monitor \
  --calibration results/reproducibility/macrostate_calibration.json \
  --output results/pilot/monitor_formulation_comparison.json \
  --seeds 501,502,503 --horizon 18
