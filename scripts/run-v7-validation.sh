#!/usr/bin/env bash
set -euo pipefail
REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${REPOSITORY}/results/complexity_entropic_coordination_v7"
PYTHON_BIN="${THERMO_V7_PYTHON:-${REPOSITORY}/.venv/bin/python}"
"${PYTHON_BIN}" - "$RESULTS" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "manifests" / "stage_disposition.json"
value = json.loads(path.read_text()) if path.exists() else {}
if not value.get("validation_unlocked", False):
    raise SystemExit("V7 validation is prospectively locked; no episodes were started")
raise SystemExit("validation unlocked but the frozen validation runner must be generated after learned-agent qualification")
PY
