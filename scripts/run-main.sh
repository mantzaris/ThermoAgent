#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$repo_dir/results/reproducibility/protocol_freeze.json" ]] || {
  echo "Main evaluation requires ./scripts/freeze-protocol.sh first." >&2
  exit 2
}
cd "$repo_dir"
.venv/bin/python -m thermoagent verify-protocol --root . \
  --freeze results/reproducibility/protocol_freeze.json >/dev/null
exec "$repo_dir/scripts/run-sweep.sh" "$repo_dir/configs/main.yaml"
