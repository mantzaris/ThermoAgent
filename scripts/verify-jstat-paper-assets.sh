#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$repository/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin=python3
cd "$repository"

"$python_bin" - <<'PY'
import json
from pathlib import Path

from thermoagent.statmech_llm.validation import validate_publication

result = validate_publication(Path.cwd())
print(json.dumps(result, indent=2, sort_keys=True))
if result["status"] != "passed":
    raise SystemExit(1)
PY

pdfs=(paper/JSTAT/main.pdf paper/JSTAT/figures/*.pdf)
if [[ "${#pdfs[@]}" -ne 15 ]]; then
  echo "Expected the manuscript and 14 publication figures" >&2
  exit 1
fi

for pdf in "${pdfs[@]}"; do
  pdfinfo "$pdf" >/dev/null
  if [[ -z "$(pdftotext "$pdf" -)" ]]; then
    echo "PDF has no extractable text: $pdf" >&2
    exit 1
  fi
  if pdffonts "$pdf" | tail -n +3 | awk 'NF && $(NF-4) != "yes" {exit 1}'; then
    :
  else
    echo "PDF has an unembedded font: $pdf" >&2
    exit 1
  fi
done

echo "Verified immutable tables, 15 equations, 14 source-backed figures, and PDF assets."
