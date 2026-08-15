#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/results/human_operator_v4/logs"
(cd "$repo_dir" && python3 -m thermoagent.v4_cli --root "$repo_dir" development) \
  2>&1 | tee "$repo_dir/results/human_operator_v4/logs/development.log"
