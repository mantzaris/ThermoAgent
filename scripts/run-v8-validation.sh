#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
if [ -f results/entropy_triggered_belief_monitoring_v8/validation/execution_summary.json ]; then
  echo "V8 validation already has a completion manifest; refusing a second execution" >&2
  exit 2
fi
"$python_exec" - <<'PY'
import json
from pathlib import Path
path = Path("results/entropy_triggered_belief_monitoring_v8/development_final/combined_progression_gates.json")
if not path.exists() or not json.loads(path.read_text())["validation_unlocked"]:
    raise SystemExit("V8 validation is prospectively locked by development gates")
PY
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  pilots --config v8_validation_frozen.yaml --stage validation
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  analyze-primary --stage validation
