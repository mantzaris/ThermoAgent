#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
test -f results/entropy_triggered_belief_monitoring_v8/training/training_summary.json
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  pilots --config v8_seed_stability_frozen.yaml --stage seed_stability
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  analyze-seed-stability --stage seed_stability
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  combine-development-gates
