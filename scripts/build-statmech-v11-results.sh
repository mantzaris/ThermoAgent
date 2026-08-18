#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m thermoagent.statmech_llm_v11.cli build-results
"${PYTHON_BIN}" -m thermoagent.statmech_llm_v11.cli generate-figures
"${PYTHON_BIN}" -m thermoagent.statmech_llm_v11.cli validate-pdfs
