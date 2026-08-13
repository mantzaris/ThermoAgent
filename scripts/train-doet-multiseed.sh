#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
trigger_config="results/entropy_triggered_v2/protocol/selected_trigger.json"
if [[ ! -f "$trigger_config" ]]; then
  echo "Missing validation-selected trigger: $trigger_config" >&2
  echo "Run scripts/run-doet-validation.sh first." >&2
  exit 2
fi
"$python_bin" -m thermoagent train-doet-multiseed \
  --results results/entropy_triggered_v2 \
  --seeds "${THERMO_RL_SEEDS:-7301,7302,7303,7304,7305}" \
  --episodes "${THERMO_TRAIN_EPISODES:-192}" \
  --calibration results/reproducibility/macrostate_calibration.json \
  --trigger-config "$trigger_config"
