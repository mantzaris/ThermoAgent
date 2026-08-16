#!/usr/bin/env bash
set -euo pipefail
REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${REPOSITORY}/results/complexity_entropic_coordination_v7"
PYTHON_BIN="${THERMO_V7_PYTHON:-$(command -v python3)}"
"${PYTHON_BIN}" -m thermoagent validate-pdfs --results "${RESULTS}"
