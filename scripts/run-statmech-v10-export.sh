#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
"$python_exec" -m thermoagent.statmech_llm.cli summary --repository "$repository_dir"
exec "$python_exec" -m thermoagent.statmech_llm.cli export --repository "$repository_dir"
