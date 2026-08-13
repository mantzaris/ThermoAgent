#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"
freeze_path="results/entropy_triggered_v2/protocol/holdout_freeze.json"
[[ ! -e "$freeze_path" ]] || {
  echo "The v2 holdout is already frozen; refusing to replace $freeze_path" >&2
  exit 2
}

required=(
  configs/entropy_trigger_validation.yaml
  configs/entropy_trigger_holdout_locked.yaml
  notes/14_entropy_trigger_protocol.md
  results/reproducibility/macrostate_calibration.json
  results/entropy_triggered_v2/calibration/trigger_nominal_calibration.json
  results/entropy_triggered_v2/validation/trigger_selection.json
  results/entropy_triggered_v2/validation/budget_matched_controls.json
  results/entropy_triggered_v2/validation/selected_trigger_pairs.csv
  results/entropy_triggered_v2/manifests/profile_v2_sweep.json
  results/entropy_triggered_v2/manifests/validation_sweep.json
  results/entropy_triggered_v2/logs/setup/model_smoke.json
  results/entropy_triggered_v2/protocol/selected_trigger.json
  results/entropy_triggered_v2/protocol/holdout_design.csv
  results/entropy_triggered_v2/protocol/holdout_design_manifest.json
  results/entropy_triggered_v2/protocol/power_precision_analysis.json
  results/entropy_triggered_v2/protocol/LOCKED_PROTOCOL.md
  results/entropy_triggered_v2/protocol/runtime_budget_fallback_preregistration.json
  results/entropy_triggered_v2/protocol/h4_evaluability_audit.json
  results/entropy_triggered_v2/training/seed_manifest.csv
  results/entropy_triggered_v2/training/training_attempts.csv
  results/entropy_triggered_v2/training/checkpoint_selection.csv
  results/entropy_triggered_v2/training/training_manifest.json
  results/entropy_triggered_v2/reproducibility/validation_replay_report.json
  results/entropy_triggered_v2/reproducibility/execution_source.json
  results/entropy_triggered_v2/reproducibility/source_transition_equivalence.json
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || {
    echo "Cannot freeze: missing $path" >&2
    exit 2
  }
done
"$python_bin" - \
  results/entropy_triggered_v2/manifests/validation_sweep.json \
  results/entropy_triggered_v2/reproducibility/validation_replay_report.json \
  results/entropy_triggered_v2/training/training_manifest.json <<'PY'
import json
import sys
from pathlib import Path

validation = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
replay = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
training = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
planned = int(validation.get("episodes_planned", -1))
complete = int(validation.get("episodes_complete", -1))
failed = int(validation.get("episodes_failed", -1))
if planned != 144 or complete != planned or failed != 0:
    raise SystemExit(
        "Cannot freeze: validation must complete all 144 episodes without failure"
    )
if (
    int(replay.get("episodes_checked", -1)) != planned
    or int(replay.get("episodes_passed", -1)) != planned
    or any(not row.get("replay_passed") for row in replay.get("records", []))
):
    raise SystemExit(
        "Cannot freeze: every validation event ledger must replay exactly"
    )
if (
    training.get("status") != "complete"
    or int(training.get("planned_trainings", -1)) != 15
    or int(training.get("completed_trainings", -1)) != 15
    or int(training.get("failed_trainings", -1)) != 0
):
    raise SystemExit(
        "Cannot freeze: all fifteen fixed-budget training runs must complete"
    )
PY
"$python_bin" - results/entropy_triggered_v2/reproducibility/execution_source.json <<'PY'
import json
import sys
from pathlib import Path

from thermoagent.experiments import source_checksum

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record.get("dirty") is not False:
    raise SystemExit(
        "Cannot freeze: execution provenance must come from a clean committed tree"
    )
if record.get("branch") != "entropy-triggered-communication":
    raise SystemExit(
        "Cannot freeze: unexpected source branch %r" % record.get("branch")
    )
observed = source_checksum(Path.cwd())
if record.get("source_checksum") != observed:
    raise SystemExit(
        "Cannot freeze: deployed source differs from execution provenance"
    )
PY
"$python_bin" - \
  results/entropy_triggered_v2/reproducibility/execution_source.json \
  results/entropy_triggered_v2/reproducibility/source_transition_equivalence.json \
  results/entropy_triggered_v2/manifests/validation_sweep.json <<'PY'
import json
import sys
from pathlib import Path

provenance = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
transition = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
validation = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if transition.get("status") != "passed":
    raise SystemExit("Cannot freeze: source-transition equivalence did not pass")
if transition.get("current_source", {}).get("source_checksum") != provenance.get(
    "source_checksum"
):
    raise SystemExit("Cannot freeze: source-transition report is stale")
if transition.get("validation_source", {}).get("source_checksum") != validation.get(
    "source_checksum"
):
    raise SystemExit(
        "Cannot freeze: source-transition report does not match validation source"
    )
PY
mapfile -t checkpoints < <(
  find results/entropy_triggered_v2/checkpoints -maxdepth 1 -type f \
    -name 'coordination_*_seed*.pt' | sort
)
if (( ${#checkpoints[@]} != 15 )); then
  echo "Cannot freeze: expected 15 learned checkpoints, found ${#checkpoints[@]}" >&2
  exit 2
fi

"$python_bin" -m pytest -q
"$python_bin" -m thermoagent freeze-protocol \
  --output "$freeze_path" --root . \
  "${required[@]}" "${checkpoints[@]}"
"$python_bin" -m thermoagent verify-protocol \
  --root . --freeze "$freeze_path" >/dev/null
echo "Frozen DOET holdout protocol at $freeze_path"
