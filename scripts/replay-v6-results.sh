#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p "$repo_dir/results/generalized_entropic_consensus_v6/logs"
exec 9>"$repo_dir/results/generalized_entropic_consensus_v6/logs/v6-writer.lock"
flock -n 9 || { echo "Another V6 writer is active; refusing concurrent execution." >&2; exit 75; }
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"

"$python_bin" - <<'PY'
from pathlib import Path
from thermoagent.v6_replay import replay_all
print(replay_all(Path("results/generalized_entropic_consensus_v6")))
PY
