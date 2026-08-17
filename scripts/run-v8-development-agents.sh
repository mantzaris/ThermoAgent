#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
test -f results/entropy_triggered_belief_monitoring_v8/training/checkpoints/v8-ippo-five-seed-ensemble.json.gz
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  pilots --config v8_development_agent_frozen.yaml --stage development_agent
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  analyze-primary --stage development_agent
