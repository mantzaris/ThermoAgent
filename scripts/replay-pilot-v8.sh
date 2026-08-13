#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
exec "$python_bin" -m thermoagent replay \
  --results results --stages pilot \
  --run-id-contains paired_nominal_v8 \
  --run-id-contains paired_compound_v8 \
  --report-name pilot_v8_replay_report.json
