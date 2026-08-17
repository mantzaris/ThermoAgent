#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
exec "$python_exec" -c \
  'from pathlib import Path; import json; from thermoagent.v8_protocol import close_v8_development_no_go; root=Path.cwd(); results=root/"results/entropy_triggered_belief_monitoring_v8"; print(json.dumps(close_v8_development_no_go(root, results), indent=2, sort_keys=True))'
