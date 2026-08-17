#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-/usr/bin/python3}"
"$python_exec" -m thermoagent.statmech.cli analyze --repository "$repository_dir"
"$python_exec" -m thermoagent.statmech.cli figures --repository "$repository_dir"
"$python_exec" -m thermoagent.statmech.cli qa --repository "$repository_dir"
