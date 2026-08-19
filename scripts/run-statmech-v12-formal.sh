#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
: "${THERMO_V12_ENABLE_QWEN:?Set THERMO_V12_ENABLE_QWEN=1 only on the existing authorized Pod}"
export THERMO_V12_ARTIFACT_ROOT="${THERMO_V12_ARTIFACT_ROOT:-/workspace/ThermoAgent-v12-artifacts}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
"$python_bin" -m thermoagent.statmech_llm_v12.cli formal
