#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${1:-/tmp/thermoagent-preflight}"
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
mkdir -p "$output_dir"
cd "$repo_dir"
"$python_bin" -m thermoagent sweep --config configs/preflight_mock.yaml --results "$output_dir" --root .
"$python_bin" -m thermoagent sweep --config configs/preflight_ablations_mock.yaml --results "$output_dir" --root .
"$python_bin" -m thermoagent sweep --config configs/preflight_holdout_mock.yaml --results "$output_dir" --root .
"$python_bin" -m thermoagent replay --results "$output_dir" --stages main ablations holdout
"$python_bin" -m thermoagent analyze --results "$output_dir"
"$python_bin" -m thermoagent figures --results "$output_dir"
"$python_bin" -m thermoagent validate-pdfs --results "$output_dir"
