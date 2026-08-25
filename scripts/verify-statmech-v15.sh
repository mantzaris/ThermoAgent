#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/.venv/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python3
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
export THERMO_V15_ARTIFACT_ROOT="${THERMO_V15_ARTIFACT_ROOT:-/workspace/ThermoAgent-v15-artifacts}"
cd "$ROOT"
"$PYTHON_BIN" -m thermoagent.statmech_llm_v15.cli pdf-qa
if [[ "${THERMO_V15_MANUAL_PDF_QA_CONFIRMED:-0}" != "1" ]]; then
  echo "Inspect every external 300-DPI rendering and original PDF, then rerun with THERMO_V15_MANUAL_PDF_QA_CONFIRMED=1" >&2
  exit 3
fi
"$PYTHON_BIN" -m thermoagent.statmech_llm_v15.cli pdf-qa-record \
  --manual-status passed \
  --manual-notes "Original vector PDFs and external 300-DPI renderings inspected; no clipping, overlap, missing glyphs, or unreadable legends observed."
# PDF rendering changes the external artifact tree after the first report was
# assembled. Refresh the compact report and repository manifest only after the
# digest-bound manual disposition so the final external checksum covers those
# QA renderings. This does not rebuild or alter any PDF.
"$PYTHON_BIN" -m thermoagent.statmech_llm_v15.cli report >/dev/null
"$PYTHON_BIN" -m thermoagent.statmech_llm_v15.cli verify
if [[ -d .git ]]; then
  git diff --check
fi
