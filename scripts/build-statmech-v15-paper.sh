#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1787443941}"
export FORCE_SOURCE_DATE="${FORCE_SOURCE_DATE:-1}"
export TZ="${TZ:-UTC}"
cd "$ROOT/paper/jstat_v15"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
if [[ -f supplement.tex ]]; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
  latexmk -c supplement.tex
fi
latexmk -c main.tex
