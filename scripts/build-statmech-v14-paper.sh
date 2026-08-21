#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/paper/jstat_v14"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -c

