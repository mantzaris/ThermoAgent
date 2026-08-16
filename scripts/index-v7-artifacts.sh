#!/usr/bin/env bash
set -euo pipefail
REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${THERMO_V7_PYTHON:-${REPOSITORY}/.venv/bin/python}"
"${PYTHON_BIN}" -m thermoagent.v7_cli --repository "${REPOSITORY}" index-artifacts
