#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export THERMO_CACHE="${THERMO_CACHE:-/workspace/.cache/thermoagent}"
mkdir -p "$repo_dir/results/human_operator_v4/logs"
(cd "$repo_dir" && "$repo_dir/.venv/bin/python" -m thermoagent.v4_cli --root "$repo_dir" real-qwen) \
  2>&1 | tee "$repo_dir/results/human_operator_v4/logs/real_qwen.log"
