#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
export THERMO_V10_ENABLE_QWEN=1
exec "$python_exec" -m thermoagent.statmech_llm.cli qwen-pilot --repository "$repository_dir"
