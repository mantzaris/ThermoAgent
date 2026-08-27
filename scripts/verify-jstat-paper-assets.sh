#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT/paper/JSTAT"
FIGURE_DIR="$PAPER_DIR/figures"
MANIFEST="$FIGURE_DIR/figure_manifest.csv"
MAIN_TEX="$PAPER_DIR/main.tex"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing JSTAT figure manifest: $MANIFEST" >&2
  exit 1
fi

expected_header="printed_figure,local_filename,semantic_label,canonical_source,local_sha256,canonical_sha256,status"
read -r header < "$MANIFEST"
if [[ "$header" != "$expected_header" ]]; then
  echo "Unexpected JSTAT figure manifest header" >&2
  exit 1
fi

declare -A expected_files=()
count=0
while IFS=, read -r printed_figure local_filename semantic_label canonical_source local_sha256 canonical_sha256 status; do
  [[ -n "$printed_figure" ]] || continue
  count=$((count + 1))
  if [[ "$printed_figure" != "$count" ]]; then
    echo "Manifest figure order is not sequential at row $count" >&2
    exit 1
  fi
  expected_files["$local_filename"]=1
  local_path="$FIGURE_DIR/$local_filename"
  canonical_path="$ROOT/$canonical_source"
  if [[ ! -f "$local_path" ]]; then
    echo "Missing local publication figure: $local_path" >&2
    exit 1
  fi
  if [[ ! -f "$canonical_path" ]]; then
    echo "Missing canonical publication figure: $canonical_path" >&2
    exit 1
  fi
  observed_local_sha256="$(sha256sum "$local_path" | cut -d' ' -f1)"
  observed_canonical_sha256="$(sha256sum "$canonical_path" | cut -d' ' -f1)"
  if [[ "$observed_local_sha256" != "$local_sha256" ]]; then
    echo "Local hash mismatch for $local_filename" >&2
    exit 1
  fi
  if [[ "$observed_canonical_sha256" != "$canonical_sha256" ]]; then
    echo "Canonical hash mismatch for $canonical_source" >&2
    exit 1
  fi
  if [[ "$observed_local_sha256" != "$observed_canonical_sha256" || "$status" != "match" ]]; then
    echo "Local and canonical figures do not match for $local_filename" >&2
    exit 1
  fi
  reference_count="$(rg -F -c "\\resultfigure{$local_filename}" "$MAIN_TEX" || true)"
  if [[ "${reference_count:-0}" != "1" ]]; then
    echo "Expected one main.tex reference to $local_filename; found ${reference_count:-0}" >&2
    exit 1
  fi
  if ! rg -F -q "{$semantic_label}" "$MAIN_TEX"; then
    echo "Missing semantic label $semantic_label in main.tex" >&2
    exit 1
  fi
done < <(tail -n +2 "$MANIFEST")

if [[ "$count" != "14" ]]; then
  echo "Expected 14 manifest figures; found $count" >&2
  exit 1
fi

while IFS= read -r local_path; do
  local_filename="${local_path##*/}"
  if [[ -z "${expected_files[$local_filename]+present}" ]]; then
    echo "Unexpected publication figure PDF: $local_path" >&2
    exit 1
  fi
done < <(find "$FIGURE_DIR" -maxdepth 1 -type f -name '*.pdf' -print | sort)

if rg -n '(\.\./\.\./results|collective_agent_statmech_v15/figures/pdf|paper/jstat_v15)' "$MAIN_TEX"; then
  echo "main.tex retains an external or obsolete manuscript dependency" >&2
  exit 1
fi

if ! rg -F -q '\newcommand{\figroot}{figures}' "$MAIN_TEX"; then
  echo "main.tex does not define the local figure root" >&2
  exit 1
fi

echo "Verified 14 self-contained JSTAT publication figures and canonical hashes."
