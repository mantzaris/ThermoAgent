#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"
log_path="$repo_dir/results/human_operator_v3/logs/setup/complete_test_suite.log"
mkdir -p "$(dirname "$log_path")"
set +e
"$python_bin" -m pytest -q 2>&1 | tee "$log_path"
status="${PIPESTATUS[0]}"
set -e
(( status == 0 )) || exit "$status"
echo "Complete v1/v2/v3 suite passed; log: $log_path"
