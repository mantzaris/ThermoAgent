#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/.venv/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python3
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
export THERMOAGENT_ARTIFACT_ROOT="${THERMOAGENT_ARTIFACT_ROOT:-/workspace/ThermoAgent-JSTAT-artifacts}"
cd "$ROOT"
scripts/run-tests.sh
scripts/verify-source-checksum.py
scripts/verify-jstat-paper-assets.sh
if [[ -d .git ]]; then
  git diff --check
fi
