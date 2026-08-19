#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
artifact_root="${THERMO_V12_ARTIFACT_ROOT:-/tmp/ThermoAgent-v12-artifacts}"
mkdir -p "$artifact_root/tests"
"$python_bin" -m pytest tests/statmech_v10 tests/statmech_v11 tests/statmech_v12 \
  --import-mode=importlib --junitxml="$artifact_root/tests/focused-junit.xml" "$@"
"$python_bin" scripts/analyze-statmech-v12-repair1.py --self-test
