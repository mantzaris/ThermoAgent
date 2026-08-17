#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
paper_dir="$repository_dir/paper/jstat_v10"
cd "$paper_dir"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdffonts main.pdf
