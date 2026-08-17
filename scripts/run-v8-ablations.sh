#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
if [ -f results/entropy_triggered_belief_monitoring_v8/ablations/execution_summary.json ]; then
  echo "V8 ablations already have a completion manifest; refusing a second execution" >&2
  exit 2
fi
test -f results/entropy_triggered_belief_monitoring_v8/protocol/v8_frozen_protocol.json
test -f results/entropy_triggered_belief_monitoring_v8/training/checkpoints/v8-ippo-five-seed-ensemble.json.gz
exec "$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  pilots --config v8_ablations_frozen.yaml --stage ablations
