#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$repo_dir/results/human_operator_v5"
mkdir -p "$results_dir/logs"
cd "$repo_dir"
python3 -c '
from pathlib import Path
from thermoagent.v5_reporting import build_v5_reporting
print(build_v5_reporting(Path(".").resolve()))
' 2>&1 | tee "$results_dir/logs/report_build.log"
# The first pass indexes the report log while `tee` is still open. Rebuild the
# checksum index after the log is closed so every recorded digest is final.
python3 -c '
from pathlib import Path
from thermoagent.v5_reporting import build_index
print(build_index(Path("results/human_operator_v5")))
'
