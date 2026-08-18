#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m thermoagent.statmech_llm_v11.cli freeze-formal
"${PYTHON_BIN}" -m thermoagent.statmech_llm_v11.cli estimate-formal
"${PYTHON_BIN}" -m thermoagent.statmech_llm_v11.cli run-formal
"${PYTHON_BIN}" -m thermoagent.statmech_llm_v11.cli analyze-formal
