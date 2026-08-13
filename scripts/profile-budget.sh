#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
hourly_rates="${THERMO_GPU_HOURLY_RATES:-0.34,0.69}"
exec "$python_bin" -m thermoagent profile-budget \
  --results results \
  --output results/reproducibility/prelaunch_budget.json \
  --hourly-rates "$hourly_rates" \
  --configs configs/main.yaml configs/ablations.yaml configs/holdout.yaml
