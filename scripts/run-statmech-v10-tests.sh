#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/thermoagent-v10-mpl}"
artifact_dir="${THERMO_V10_ARTIFACT_ROOT:-/tmp/ThermoAgent-v10-artifacts}"
mkdir -p "$artifact_dir/tests"
"$python_exec" -m pytest -q --import-mode=importlib tests/statmech_v10 tests/statmech_v9 --junitxml="$artifact_dir/tests/junit.xml"
exec "$python_exec" -m thermoagent.statmech_llm.cli test-summary --repository "$repository_dir" --junit "$artifact_dir/tests/junit.xml"
