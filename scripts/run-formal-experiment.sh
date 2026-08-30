#!/usr/bin/env bash
set -euo pipefail

if [[ "${THERMOAGENT_ENABLE_LLM:-0}" != "1" ]]; then
  echo "set THERMOAGENT_ENABLE_LLM=1 to authorize deliberate model inference" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
artifact_root="${THERMOAGENT_ARTIFACT_ROOT:-/workspace/ThermoAgent-JSTAT-artifacts}"
export THERMOAGENT_ARTIFACT_ROOT="$artifact_root"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HOME="${HF_HOME:-/workspace/ThermoAgent-JSTAT-model-cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "$ROOT"

case "${1:-}" in
  qwen|granite)
    [[ $# -eq 1 ]] || { echo "usage: $0 qwen|granite" >&2; exit 2; }
    exec "$PYTHON_BIN" -m thermoagent.statmech_llm.cli formal --model "$1"
    ;;
  discovery)
    [[ $# -eq 1 ]] || { echo "usage: $0 discovery" >&2; exit 2; }
    export THERMOAGENT_DISCOVERY_ARTIFACT_ROOT="${THERMOAGENT_DISCOVERY_ARTIFACT_ROOT:-$artifact_root/discovery}"
    exec "$PYTHON_BIN" -m thermoagent.statmech_llm.discovery.cli formal
    ;;
  replication)
    [[ $# -eq 1 ]] || { echo "usage: $0 replication" >&2; exit 2; }
    export THERMOAGENT_REPLICATION_ARTIFACT_ROOT="${THERMOAGENT_REPLICATION_ARTIFACT_ROOT:-$artifact_root/replication}"
    exec "$PYTHON_BIN" -m thermoagent.statmech_llm.replication.cli formal
    ;;
  corrected-quench)
    [[ $# -eq 1 ]] || { echo "usage: $0 corrected-quench" >&2; exit 2; }
    export THERMOAGENT_CORRECTED_QUENCH_ARTIFACT_ROOT="${THERMOAGENT_CORRECTED_QUENCH_ARTIFACT_ROOT:-$artifact_root/corrected_quench}"
    exec "$PYTHON_BIN" -m thermoagent.statmech_llm.corrected_quench.cli formal
    ;;
  *)
    echo "usage: $0 qwen|granite|discovery|replication|corrected-quench" >&2
    exit 2
    ;;
esac
