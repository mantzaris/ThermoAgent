#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"
if [[ $# -gt 0 ]]; then
  episode="$1"
  shift
else
  episode="$(find "$repo_dir/results/generalized_entropic_consensus_v6/raw" \
    -path '*development_dynamic*' -name 'episode.json.gz' -print -quit)"
fi
if [[ -z "${episode:-}" || ! -f "$episode" ]]; then
  echo "No populated V6 replay episode exists yet; run the unlocked development workflow first." >&2
  exit 3
fi
exec "$python_bin" -m thermoagent.dashboard --episode "$episode" "$@"
