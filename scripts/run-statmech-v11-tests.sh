#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m pytest -q tests/statmech_v11 tests/statmech_v10
