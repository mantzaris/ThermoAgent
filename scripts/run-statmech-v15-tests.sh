#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/.venv/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python3
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
cd "$ROOT"
for suite in statmech_v10 statmech_v11 statmech_v12 statmech_v13 statmech_v14 statmech_v15; do
  "$PYTHON_BIN" -m pytest --import-mode=importlib -q "tests/$suite"
done
