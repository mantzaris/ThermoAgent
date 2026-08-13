#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export THERMO_CACHE="${THERMO_CACHE:-/workspace/.cache/thermoagent}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/thermoagent-pycache}"
output_dir="${THERMO_SMOKE_OUTPUT_DIR:-results/smoke/stage1}"
exec .venv/bin/python -m thermoagent agentic-smoke \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --output-dir "$output_dir"
