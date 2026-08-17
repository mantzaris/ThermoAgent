#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  pilots --config v8_development_final_repaired.yaml --stage development_final
"$python_exec" -m thermoagent.v8_cli --repository "$repository_dir" \
  analyze-final-development --stage development_final --bootstrap-replicates 10000
