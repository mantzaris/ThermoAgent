#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3)"
fi

"$python_bin" -c '
from pathlib import Path
from thermoagent.v6_v5_reanalysis import run_v5_abstention_reanalysis
run_v5_abstention_reanalysis(Path.cwd(), Path("results/generalized_entropic_consensus_v6"))
'

echo "V5 abstention reanalysis written under results/generalized_entropic_consensus_v6/v5_reanalysis"
