#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/thermoagent-v8-matplotlib"
mkdir -p "$MPLCONFIGDIR"
exec "$python_exec" -c \
  'from pathlib import Path; import json; from thermoagent.v8_figures import generate_v8_figures; print(json.dumps(generate_v8_figures(Path("results/entropy_triggered_belief_monitoring_v8")), indent=2))'
