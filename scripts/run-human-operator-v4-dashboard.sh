#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if (( $# == 0 )); then
  echo "Usage: $0 --episode results/human_operator_v4/raw/<stage>/<run>/episode.json [--port 8765]" >&2
  exit 2
fi
exec python3 -m thermoagent.dashboard "$@"
