#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$repository/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin=python3
export THERMOAGENT_ARTIFACT_ROOT="${THERMOAGENT_ARTIFACT_ROOT:-/workspace/ThermoAgent-JSTAT-artifacts}"
cd "$repository"
case "${1:-cross-model}" in
  cross-model)
    exec "$python_bin" -m thermoagent.statmech_llm.cli replay
    ;;
  discovery)
    export THERMOAGENT_DISCOVERY_ARTIFACT_ROOT="${THERMOAGENT_DISCOVERY_ARTIFACT_ROOT:-$THERMOAGENT_ARTIFACT_ROOT/discovery}"
    exec "$python_bin" -m thermoagent.statmech_llm.discovery.cli replay
    ;;
  replication)
    export THERMOAGENT_REPLICATION_ARTIFACT_ROOT="${THERMOAGENT_REPLICATION_ARTIFACT_ROOT:-$THERMOAGENT_ARTIFACT_ROOT/replication}"
    exec "$python_bin" -m thermoagent.statmech_llm.replication.cli replay
    ;;
  corrected-quench)
    export THERMOAGENT_CORRECTED_QUENCH_ARTIFACT_ROOT="${THERMOAGENT_CORRECTED_QUENCH_ARTIFACT_ROOT:-$THERMOAGENT_ARTIFACT_ROOT/corrected_quench}"
    exec "$python_bin" -m thermoagent.statmech_llm.corrected_quench.cli replay
    ;;
  *)
    echo "usage: $0 [cross-model|discovery|replication|corrected-quench]" >&2
    exit 2
    ;;
esac
