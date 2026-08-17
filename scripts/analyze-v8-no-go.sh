#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
exec "$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  analyze-no-go --stage hysteresis_repair_pilot_v3 --bootstrap-replicates 10000
