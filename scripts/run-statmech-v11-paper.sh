#!/usr/bin/env bash
set -euo pipefail
paper_directory="paper/jstat_v11"
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd "${paper_directory}/main.tex"
pdfinfo "${paper_directory}/main.pdf"
pdffonts "${paper_directory}/main.pdf"
pdftotext "${paper_directory}/main.pdf" /dev/null
