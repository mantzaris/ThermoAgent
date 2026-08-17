#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
test -f results/entropy_triggered_belief_monitoring_v8/protocol/v8_frozen_protocol.json
exec "$python_exec" -m thermoagent.v8_cli \
  --repository "$repository_dir" train-multiseed \
  --seeds 88201,88202,88203,88204,88205 --episodes 18 --update-epochs 4
