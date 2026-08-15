#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export THERMO_CACHE="${THERMO_CACHE:-/workspace/.cache/thermoagent}"
mkdir -p "$repo_dir/results/human_operator_v5/logs"
cd "$repo_dir"
"$repo_dir/.venv/bin/python" -c '
from pathlib import Path
from thermoagent.v5_qwen import run_real_qwen_qualification
root = Path(".").resolve()
print(run_real_qwen_qualification(root, root / "results" / "human_operator_v5"))
' 2>&1 | tee "$repo_dir/results/human_operator_v5/logs/real_qwen_qualification.log"
