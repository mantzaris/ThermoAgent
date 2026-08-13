#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python3 -m thermoagent.doet_monitoring \
  --results-root results \
  --output-root results/entropy_triggered_v2

for name in monitoring_baseline_comparison entropy_incremental_value; do
  pdf="results/entropy_triggered_v2/figures/pdf/${name}.pdf"
  preview="results/entropy_triggered_v2/figures/previews/${name}.png"
  pdfinfo "$pdf" >/dev/null
  pdffonts "$pdf" >/dev/null
  pdftoppm -f 1 -singlefile -png -r 180 "$pdf" "${preview%.png}" >/dev/null 2>&1
done
