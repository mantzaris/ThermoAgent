#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$repo_dir/results/human_operator_v4"
mkdir -p "$results_dir/logs"
(cd "$repo_dir" && python3 -m thermoagent.v4_cli --root "$repo_dir" report) \
  2>&1 | tee "$results_dir/logs/reporting.log"
# The report command must finish before its own log is complete. Re-index once
# more afterward so the indexed reporting-log checksum is stable and verifiable.
(cd "$repo_dir" && python3 -m thermoagent.v4_cli --root "$repo_dir" index) \
  >/dev/null
