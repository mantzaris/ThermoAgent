#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FROZEN_ROOT="${THERMO_V15_FROZEN_ROOT:-/workspace/ThermoAgent-v15-frozen-b309f0ab}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/ThermoAgent/.venv/bin/python}"
export THERMO_V15_ARTIFACT_ROOT="${THERMO_V15_ARTIFACT_ROOT:-/workspace/ThermoAgent-v15-reconstruction-b309f0ab}"
export THERMO_V15_ANALYSIS_WORKERS="${THERMO_V15_ANALYSIS_WORKERS:-16}"

EXPECTED_COMMIT="b309f0ab76cb24377de5872eebc811582af1f43f"
REFERENCE_RESULT="$THERMO_V15_ARTIFACT_ROOT/reproducibility/committed_reference_result"
REFERENCE_IDENTITY="$THERMO_V15_ARTIFACT_ROOT/reproducibility/committed_reference_identity.json"
COMPARISON="$THERMO_V15_ARTIFACT_ROOT/reproducibility/reconstructed_vs_committed.json"

[[ -x "$PYTHON_BIN" ]] || {
  echo "pinned V15 Python environment is unavailable" >&2
  exit 2
}
[[ -d "$FROZEN_ROOT/.git" ]] || {
  echo "clean frozen checkout is unavailable" >&2
  exit 2
}
[[ "$(git -C "$FROZEN_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "clean frozen checkout is at the wrong commit" >&2
  exit 2
}
git -C "$FROZEN_ROOT" diff --quiet -- \
  configs/statmech_v15 thermoagent/statmech_llm_v15 tests/statmech_v15 scripts || {
  echo "disposable frozen checkout has scientific-source modifications" >&2
  exit 2
}
[[ -f "$THERMO_V15_ARTIFACT_ROOT/formal/completion.json" ]] || {
  echo "formal reconstruction has not completed" >&2
  exit 2
}
cd "$ROOT"
# Snapshot the committed package before the disposable frozen checkout is used
# for replay and its original analysis.  A failed comparison stops before the
# review working tree's repository-facing result package is regenerated.  On
# retry, reuse the immutable external snapshot only after its tree digest and
# provenance identity verify; never copy a previously analyzed disposable tree
# over the reference.
if [[ ! -e "$REFERENCE_RESULT" ]]; then
  mkdir -p "$(dirname "$REFERENCE_RESULT")"
  cp -a "$FROZEN_ROOT/results/collective_agent_statmech_v15" "$REFERENCE_RESULT"
  "$PYTHON_BIN" - "$REFERENCE_RESULT" "$REFERENCE_IDENTITY" <<'PY'
import json
import sys
from pathlib import Path
from thermoagent.statmech_llm_v15.workflow import atomic_json, tree_digest

reference, identity = map(Path, sys.argv[1:])
atomic_json(
    {
        "commit": "b309f0ab76cb24377de5872eebc811582af1f43f",
        "source": "committed result package snapshotted before reconstruction analysis",
        "tree": tree_digest(reference),
    },
    identity,
)
PY
else
  [[ -f "$REFERENCE_IDENTITY" ]] || {
    echo "reference snapshot identity is missing; preserve and inspect it" >&2
    exit 2
  }
  "$PYTHON_BIN" - "$REFERENCE_RESULT" "$REFERENCE_IDENTITY" <<'PY'
import json
import sys
from pathlib import Path
from thermoagent.statmech_llm_v15.workflow import tree_digest

reference, identity_path = map(Path, sys.argv[1:])
identity = json.loads(identity_path.read_text(encoding="utf-8"))
if identity.get("commit") != "b309f0ab76cb24377de5872eebc811582af1f43f":
    raise RuntimeError("committed reference snapshot has the wrong identity")
if identity.get("tree") != tree_digest(reference):
    raise RuntimeError("committed reference snapshot changed after creation")
PY
fi

cd "$FROZEN_ROOT"
"$PYTHON_BIN" -m thermoagent.statmech_llm_v15.cli replay
"$PYTHON_BIN" -m thermoagent.statmech_llm_v15.cli analyze

cd "$ROOT"
"$PYTHON_BIN" scripts/compare-statmech-v15-reconstruction.py \
  --reference "$REFERENCE_RESULT" \
  --reconstructed "$FROZEN_ROOT/results/collective_agent_statmech_v15" \
  --output "$COMPARISON"

# Only a passing deterministic scientific comparison authorizes rebuilding the
# current V15 result package with the post-reconstruction descriptive extension.
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from thermoagent.statmech_llm_v15.analysis import analyze_formal

analyze_formal(Path.cwd())
PY

echo "V15 replay, committed comparison, and extended analysis completed"
