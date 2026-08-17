#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/thermoagent-v10-mpl}"
"$python_exec" -m thermoagent.statmech_llm.cli analyze --repository "$repository_dir"
"$python_exec" -m thermoagent.statmech_llm.cli figures --repository "$repository_dir"
"$python_exec" -m thermoagent.statmech_llm.cli qa --repository "$repository_dir"
