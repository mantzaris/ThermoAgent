#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"

"$python_bin" - <<'PY'
from pathlib import Path
from thermoagent.v6_artifacts import build_index, verify_artifacts
root = Path("results/generalized_entropic_consensus_v6")
print(build_index(root))
print(verify_artifacts(root))
PY
