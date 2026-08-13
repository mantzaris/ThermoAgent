#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-python3}"
exec "$python_bin" -m pytest -q \
  tests/test_environment.py tests/test_independence.py tests/test_mechanics.py \
  tests/test_runner.py
