#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"

# The frozen rebuild proves tests, replay, aggregation, base figures, and PDF
# mechanics.  The separate wrapper then reapplies the documented post-freeze
# presentation-only layout corrections and validates/indexes the final PDFs.
./scripts/rebuild-results.sh
./results/reproducibility/tools/polish-figures.sh

printf '%s\n' \
  'Final derived artifacts rebuilt.' \
  'Manual preview inspection is intentionally not auto-certified.' \
  'After inspection, run the mark-visual-qa command documented in results/README.md.'
