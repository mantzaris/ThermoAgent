#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
results_root="$repository_dir/results/entropy_triggered_belief_monitoring_v8"

"$python_exec" - "$results_root" "$@" <<'PY'
import json
import sys
from pathlib import Path

from thermoagent.v8_compaction import (
    pack_completed_stage,
    pack_retained_incomplete_stage,
)

root = Path(sys.argv[1])
requested = list(sys.argv[2:])
stages = requested or [
    "routing_repair_pilot", "hysteresis_repair_pilot",
    "hysteresis_repair_pilot_v2", "hysteresis_repair_pilot_v3",
    "development", "development_final", "development_agent",
    "seed_stability", "validation", "holdout",
]
reports = []
for stage in stages:
    raw = root / "raw" / stage
    if not raw.exists():
        continue
    reports.append(pack_completed_stage(root, stage))
if not requested:
    retained = {
        "development_final_pre_hysteresis_invalidated": (108, 2),
        "development_final_hysteresis_suppression_invalidated": (288, 0),
    }
    for stage, (complete, partial) in retained.items():
        if (root / "raw" / stage).exists():
            reports.append(pack_retained_incomplete_stage(
                root, stage, expected_complete=complete, expected_partial=partial,
            ))
print(json.dumps(reports, indent=2, sort_keys=True))
PY
