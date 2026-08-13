#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
"$python_bin" -m thermoagent sweep \
  --config configs/entropy_trigger_validation.yaml \
  --results results/entropy_triggered_v2 \
  --root "$repo_dir"
"$python_bin" -m thermoagent select-doet-trigger \
  --results results/entropy_triggered_v2
