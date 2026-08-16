#!/usr/bin/env bash
set -euo pipefail
REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${THERMO_V7_PYTHON:-${REPOSITORY}/.venv/bin/python}"
"${PYTHON_BIN}" -m thermoagent.v7_cli --repository "${REPOSITORY}" run-development-reference
"${PYTHON_BIN}" -m thermoagent.v7_cli --repository "${REPOSITORY}" run-development-dynamic
"${PYTHON_BIN}" -m thermoagent.v7_cli --repository "${REPOSITORY}" run-development-communication
"${PYTHON_BIN}" -m thermoagent.v7_cli --repository "${REPOSITORY}" replay
"${PYTHON_BIN}" -m thermoagent.v7_cli --repository "${REPOSITORY}" evaluate-formal-development
