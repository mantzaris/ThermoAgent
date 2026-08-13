#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"
root="results/entropy_triggered_v2"
config="$root/protocol/extended_ablation_config.yaml"
design="$root/protocol/extended_ablation_design.json"
freeze="$root/protocol/ablation_freeze.json"

"$python_bin" -m thermoagent design-doet-ablations \
  --results "$root" --config "$config"
"$python_bin" - "$design" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if not record["authorized"]:
    raise SystemExit(
        "Extended ablations exceed the 35-hour cap; design retained without launch"
    )
PY
if [[ ! -f "$freeze" ]]; then
  "$python_bin" -m thermoagent freeze-protocol \
    --output "$freeze" --root . \
    "$config" "$design" \
    "$root/validation/trigger_selection.json" \
    "$root/protocol/holdout_freeze.json" \
    "$root/reproducibility/execution_source.json"
fi
"$python_bin" -m thermoagent verify-protocol --root . --freeze "$freeze" >/dev/null
"$python_bin" -m thermoagent sweep --config "$config" --results "$root" --root .
"$python_bin" -m thermoagent replay --results "$root" --stages ablations \
  --report-name ablations_replay_report.json
./scripts/analyze-doet-results.sh
./scripts/generate-doet-figures.sh
"$python_bin" -m thermoagent index --results "$root"
