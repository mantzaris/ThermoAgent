#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "qwen" && "$1" != "granite" ) ]]; then
  echo "usage: $0 qwen|granite" >&2
  exit 2
fi
if [[ "${THERMO_V15_ENABLE_LLM:-0}" != "1" ]]; then
  echo "set THERMO_V15_ENABLE_LLM=1 to authorize the frozen engineering pilot" >&2
  exit 2
fi

EXPECTED_COMMIT="b309f0ab76cb24377de5872eebc811582af1f43f"
FROZEN_ROOT="${THERMO_V15_FROZEN_ROOT:-/workspace/ThermoAgent-v15-frozen-b309f0ab}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/ThermoAgent/.venv/bin/python}"
export THERMO_V15_ARTIFACT_ROOT="${THERMO_V15_ARTIFACT_ROOT:-/workspace/ThermoAgent-v15-reconstruction-b309f0ab}"
export HF_HOME="${HF_HOME:-/workspace/ThermoAgent-v15-model-cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

[[ -x "$PYTHON_BIN" ]] || {
  echo "pinned V15 Python environment is unavailable" >&2
  exit 2
}
[[ -d "$FROZEN_ROOT/.git" ]] || {
  echo "clean frozen checkout is unavailable" >&2
  exit 2
}
[[ "$(git -C "$FROZEN_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "clean frozen checkout is at the wrong commit" >&2
  exit 2
}
[[ -z "$(git -C "$FROZEN_ROOT" status --porcelain --untracked-files=no)" ]] || {
  echo "clean frozen checkout has tracked modifications" >&2
  exit 2
}

cd "$FROZEN_ROOT"
"$PYTHON_BIN" -m thermoagent.statmech_llm_v15.cli pilot --model "$1"
