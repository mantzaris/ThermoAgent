#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
mkdir -p results/checkpoints results/logs/training
"$python_bin" -m thermoagent train-policy \
  --variant no_entropy --episodes "${THERMO_TRAIN_EPISODES:-192}" --seed 3001 \
  --calibration results/reproducibility/macrostate_calibration.json \
  --output results/checkpoints/coordination_no_entropy.pt \
  --log results/logs/training/no_entropy.jsonl
"$python_bin" -m thermoagent train-policy \
  --variant thermo --episodes "${THERMO_TRAIN_EPISODES:-192}" --seed 3001 \
  --calibration results/reproducibility/macrostate_calibration.json \
  --output results/checkpoints/coordination_thermo.pt \
  --log results/logs/training/thermo.jsonl
