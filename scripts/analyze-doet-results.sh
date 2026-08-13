#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"
"$python_bin" -m thermoagent analyze-doet \
  --results results/entropy_triggered_v2
"$python_bin" -m thermoagent report-doet \
  --results results/entropy_triggered_v2
