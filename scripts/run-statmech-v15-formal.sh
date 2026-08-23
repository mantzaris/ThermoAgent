#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "qwen" && "$1" != "granite" ) ]]; then
  echo "usage: $0 qwen|granite" >&2
  exit 2
fi
if [[ "${THERMO_V15_ENABLE_LLM:-0}" != "1" ]]; then
  echo "set THERMO_V15_ENABLE_LLM=1 to authorize the frozen existing-Pod model run" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/ThermoAgent/.venv/bin/python}"
export THERMO_V15_ARTIFACT_ROOT="${THERMO_V15_ARTIFACT_ROOT:-/workspace/ThermoAgent-v15-artifacts}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "$ROOT"
"$PYTHON_BIN" -m thermoagent.statmech_llm_v15.cli formal --model "$1"
