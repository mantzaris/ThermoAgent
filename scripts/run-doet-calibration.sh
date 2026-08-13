#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
"$python_bin" -m thermoagent calibrate-doet \
  --output results/entropy_triggered_v2/calibration \
  --nominal-seeds 5101,5102,5103,5104,5105,5106 \
  --development-seeds 5201,5202,5203 \
  --horizon 24
