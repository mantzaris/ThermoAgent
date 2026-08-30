#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$repository/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin=python3
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1787443941}"
export TZ="${TZ:-UTC}"
cd "$repository"

mode="${1:-verify}"
case "$mode" in
  verify)
    output_root="$(mktemp -d /tmp/thermoagent-figure-rebuild.XXXXXX)"
    export THERMOAGENT_PUBLICATION_OUTPUT_ROOT="$output_root"
    ;;
  --in-place)
    unset THERMOAGENT_PUBLICATION_OUTPUT_ROOT
    output_root="$repository/results/JSTAT"
    ;;
  *)
    echo "Usage: $0 [verify|--in-place]" >&2
    exit 2
    ;;
esac

"$python_bin" -m thermoagent.statmech_llm.cli figures

if [[ "$mode" == "verify" ]]; then
  expected_count="$(find results/JSTAT/source_data -maxdepth 1 -type f -name '*.csv' | wc -l)"
  rebuilt_count="$(find "$output_root/source_data" -maxdepth 1 -type f -name '*.csv' | wc -l)"
  if [[ "$expected_count" != "14" || "$rebuilt_count" != "$expected_count" ]]; then
    echo "Expected 14 canonical and rebuilt source-data tables" >&2
    exit 1
  fi
  while IFS= read -r canonical; do
    rebuilt="$output_root/source_data/$(basename "$canonical")"
    if [[ ! -f "$rebuilt" ]] || ! cmp -s "$canonical" "$rebuilt"; then
      echo "Figure source-data mismatch: $canonical" >&2
      exit 1
    fi
  done < <(find results/JSTAT/source_data -maxdepth 1 -type f -name '*.csv' | sort)
  echo "Rebuilt 14 figure datasets byte-for-byte in $output_root"
  echo "Canonical PDFs were not overwritten; use --in-place only for a deliberate rebuild."
fi

scripts/verify-jstat-paper-assets.sh
