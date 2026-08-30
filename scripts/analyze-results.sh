#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$repository/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin=python3
export THERMOAGENT_ARTIFACT_ROOT="${THERMOAGENT_ARTIFACT_ROOT:-/workspace/ThermoAgent-JSTAT-artifacts}"
export THERMOAGENT_ANALYSIS_WORKERS="${THERMOAGENT_ANALYSIS_WORKERS:-1}"
cd "$repository"
case "${1:-cross-model}" in
  cross-model)
    "$python_bin" -m thermoagent.statmech_llm.cli analyze
    "$python_bin" -m thermoagent.statmech_llm.cli surrogate
    ;;
  discovery)
    export THERMOAGENT_DISCOVERY_ARTIFACT_ROOT="${THERMOAGENT_DISCOVERY_ARTIFACT_ROOT:-$THERMOAGENT_ARTIFACT_ROOT/discovery}"
    "$python_bin" -m thermoagent.statmech_llm.discovery.cli analyze
    ;;
  replication)
    export THERMOAGENT_REPLICATION_ARTIFACT_ROOT="${THERMOAGENT_REPLICATION_ARTIFACT_ROOT:-$THERMOAGENT_ARTIFACT_ROOT/replication}"
    "$python_bin" -m thermoagent.statmech_llm.replication.cli analyze
    ;;
  corrected-quench)
    export THERMOAGENT_CORRECTED_QUENCH_ARTIFACT_ROOT="${THERMOAGENT_CORRECTED_QUENCH_ARTIFACT_ROOT:-$THERMOAGENT_ARTIFACT_ROOT/corrected_quench}"
    "$python_bin" -m thermoagent.statmech_llm.corrected_quench.cli analyze
    ;;
  *)
    echo "usage: $0 [cross-model|discovery|replication|corrected-quench]" >&2
    exit 2
    ;;
esac
