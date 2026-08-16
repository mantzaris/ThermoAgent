#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$repo_dir/results/generalized_entropic_consensus_v6"
mkdir -p "$results_dir/logs"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-$(command -v python3)}"
"$python_bin" -m thermoagent validate-pdfs --results "$results_dir" \
  2>&1 | tee "$results_dir/logs/pdf_validation.log"
