#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
# The complete historical suite includes the original PyTorch policy tests.
# The lightweight V8 virtualenv intentionally omits PyTorch; the repository's
# established system interpreter provides the compatible CPU build.  Operators
# can still override this explicitly with THERMO_PYTHON.
python_exec="${THERMO_PYTHON:-/usr/bin/python3}"
report_dir="results/entropy_triggered_belief_monitoring_v8/reproducibility"
mkdir -p "$report_dir"
"$python_exec" -m pytest -q \
  --junitxml "$report_dir/pytest_full.xml"
"$python_exec" -m pytest -q tests/test_v8_*.py \
  --junitxml "$report_dir/pytest_v8.xml"
