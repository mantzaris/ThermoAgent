#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/.venv/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python3
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
cd "$ROOT"
"$PYTHON_BIN" -m thermoagent.statmech_llm_v15.cli figures
