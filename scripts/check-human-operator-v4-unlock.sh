#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage="${1:?stage name required}"
python3 - "$repo_dir/results/human_operator_v4/development/gate_status.json" "$stage" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
stage = sys.argv[2]
if not status.get("all_required_gates_passed", False):
    failed = [name for name, value in status["gates"].items() if not value["passed"]]
    raise SystemExit(
        "ThermoHITL v4 %s is prospectively locked; failed gate(s): %s"
        % (stage, ", ".join(failed))
    )
raise SystemExit(
    "Gate report permits %s, but this stopped v4 snapshot intentionally has no later-stage runner. "
    "Create a new protocol/version rather than mutating v4." % stage
)
PY
