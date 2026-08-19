#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
export THERMO_V12_ARTIFACT_ROOT="${THERMO_V12_ARTIFACT_ROOT:-/workspace/ThermoAgent-v12-artifacts}"
"$python_bin" -m thermoagent.statmech_llm_v12.cli report
