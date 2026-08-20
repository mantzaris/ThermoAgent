#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
"$python_bin" -m pytest --import-mode=importlib -q \
  tests/statmech_v10 tests/statmech_v11 tests/statmech_v12 tests/statmech_v13
