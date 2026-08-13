#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
"$python_bin" results/reproducibility/tools/polish_figures.py --results results
"$python_bin" -m thermoagent validate-pdfs --results results
"$python_bin" -m thermoagent index --results results
