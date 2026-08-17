#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
if [ -f results/entropy_triggered_belief_monitoring_v8/holdout/execution_summary.json ]; then
  echo "V8 holdout already has a completion manifest; refusing a second execution" >&2
  exit 2
fi
"$python_exec" - <<'PY'
import json
from pathlib import Path
path = Path("results/entropy_triggered_belief_monitoring_v8/validation/primary_gate_results.json")
if not path.exists() or not json.loads(path.read_text())["progression_pass"]:
    raise SystemExit("V8 holdout is prospectively locked by validation gates")
PY
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  pilots --config v8_holdout_locked.yaml --stage holdout
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  analyze-primary --stage holdout
