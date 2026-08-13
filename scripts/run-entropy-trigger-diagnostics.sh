#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python3 -m thermoagent.doet_diagnostics \
  --results-root results \
  --output-root results/entropy_triggered_v2

pdf="results/entropy_triggered_v2/figures/pdf/original_holdout_tie_diagnostics.pdf"
preview="results/entropy_triggered_v2/figures/previews/original_holdout_tie_diagnostics.png"
pdfinfo "$pdf" >/dev/null
pdffonts "$pdf" >/dev/null
pdftoppm -f 1 -singlefile -png -r 180 "$pdf" "${preview%.png}" >/dev/null 2>&1
