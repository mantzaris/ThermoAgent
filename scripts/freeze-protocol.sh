#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
.venv/bin/python -m thermoagent freeze-protocol \
  --output results/reproducibility/protocol_freeze.json --root . \
  configs/main.yaml configs/ablations.yaml configs/holdout.yaml \
  results/reproducibility/macrostate_calibration.json \
  results/checkpoints/coordination_no_entropy.pt \
  results/checkpoints/coordination_thermo.pt \
  pyproject.toml requirements-runpod.txt thermoagent/*.py \
  scripts/freeze-protocol.sh scripts/run-sweep.sh scripts/run-main.sh \
  scripts/run-ablations.sh scripts/run-holdout.sh \
  scripts/analyze-results.sh scripts/generate-figures.sh \
  scripts/replay-results.sh scripts/validate-pdfs.sh
