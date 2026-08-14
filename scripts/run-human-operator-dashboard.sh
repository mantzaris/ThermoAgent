#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"
if (( $# == 0 )); then
  echo "Usage: $0 --live | --episode results/human_operator_v3/raw/<stage>/<run>/episode.json [--port 8765]" >&2
  exit 2
fi
exec "$python_bin" -m thermoagent.dashboard "$@"
