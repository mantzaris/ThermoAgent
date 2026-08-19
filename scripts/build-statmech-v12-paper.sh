#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${THERMO_V12_ARTIFACT_ROOT:-}" ]]; then
  if [[ -d /workspace/ThermoAgent-v12-artifacts && -w /workspace/ThermoAgent-v12-artifacts ]]; then
    export THERMO_V12_ARTIFACT_ROOT=/workspace/ThermoAgent-v12-artifacts
  else
    export THERMO_V12_ARTIFACT_ROOT=/tmp/ThermoAgent-v12-artifacts
  fi
fi
cd "$repo_dir/paper/jstat_v12"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
cd "$repo_dir"
"${PYTHON_BIN:-$repo_dir/.venv/bin/python}" -m thermoagent.statmech_llm_v12.cli pdf-qa
