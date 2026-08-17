#!/usr/bin/env bash
set -euo pipefail
REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY"
if [[ -n "${THERMO_V7_PYTHON:-}" ]]; then
  python_bin="$THERMO_V7_PYTHON"
elif [[ -x "$REPOSITORY/.venv/bin/python" ]] \
    && "$REPOSITORY/.venv/bin/python" -c 'import pytest, torch' >/dev/null 2>&1; then
  python_bin="$REPOSITORY/.venv/bin/python"
else
  python_bin="$(command -v python3)"
fi
report="$REPOSITORY/results/complexity_entropic_coordination_v7/reproducibility/pytest_v7.xml"
mkdir -p "$(dirname "$report")"
"$python_bin" -m pytest -q tests/test_v7_*.py --junitxml="$report"
