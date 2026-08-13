#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then
  echo "Usage: $0 configs/<stage>.yaml" >&2
  exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export THERMO_CACHE="${THERMO_CACHE:-/workspace/.cache/thermoagent}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/thermoagent-pycache}"
exec .venv/bin/python -m thermoagent sweep --config "$1" --results results --root .
