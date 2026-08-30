#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT/paper/JSTAT"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1787443941}"
export FORCE_SOURCE_DATE="${FORCE_SOURCE_DATE:-1}"
export TZ="${TZ:-UTC}"

expected_figures=(
  figure01_architecture.pdf
  figure02_memory_evidence_stages.pdf
  figure03_corrected_quench_time_series.pdf
  figure04_cross_model_quench.pdf
  figure05_graph_distance_correlations.pdf
  figure06_memory_controls.pdf
  figure07_confirmatory_effects.pdf
  figure08_direct_surrogate_quench.pdf
  figure09_cluster_recovery.pdf
  figure10_delayed_audit.pdf
  figure11_path_reversal_sensitivity.pdf
  figure12_prompt_balance.pdf
  figure13_surrogate_size_context.pdf
  figure14_persistence_binder.pdf
)

for figure in "${expected_figures[@]}"; do
  if [[ ! -f "$PAPER_DIR/figures/$figure" ]]; then
    echo "Missing required JSTAT publication figure: $PAPER_DIR/figures/$figure" >&2
    exit 1
  fi
done

cd "$PAPER_DIR"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
if grep -Eiq 'undefined citations|undefined references|Citation .* undefined|Reference .* undefined' main.log; then
  echo "The JSTAT build contains unresolved citations or references" >&2
  exit 1
fi
latexmk -c main.tex
rm -f main.bbl
