#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
exec "$python_exec" -c \
  'from pathlib import Path; import json; from thermoagent.v8_protocol import freeze_v8_protocol; root=Path.cwd(); print(json.dumps(freeze_v8_protocol(root, root/"results/entropy_triggered_belief_monitoring_v8"), indent=2, sort_keys=True))'
