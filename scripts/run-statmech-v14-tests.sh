#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$ROOT"
for suite in statmech_v10 statmech_v11 statmech_v12 statmech_v13 statmech_v14; do
  "$PYTHON_BIN" -m pytest -q "tests/$suite"
done
