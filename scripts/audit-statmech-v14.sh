#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/ThermoAgent/.venv/bin/python}"
export THERMO_V14_ARTIFACT_ROOT="${THERMO_V14_ARTIFACT_ROOT:-/workspace/ThermoAgent-v14-artifacts}"
export THERMO_V13_ARTIFACT_ROOT="${THERMO_V13_ARTIFACT_ROOT:-/workspace/ThermoAgent-v13-artifacts}"
export THERMO_V12_ARTIFACT_ROOT="${THERMO_V12_ARTIFACT_ROOT:-/workspace/ThermoAgent-v12-artifacts}"
export THERMO_V14_PERMUTATION_WORKERS="${THERMO_V14_PERMUTATION_WORKERS:-16}"
cd "$ROOT"
"$PYTHON_BIN" -m thermoagent.statmech_llm_v14.cli analyze
"$PYTHON_BIN" -m thermoagent.statmech_llm_v14.cli figures
"$PYTHON_BIN" -m thermoagent.statmech_llm_v14.cli report

