#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  pilots --config v8_hysteresis_repair_pilot_v3.yaml --stage hysteresis_repair_pilot_v3
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  analyze-hysteresis-repair --stage hysteresis_repair_pilot_v3
