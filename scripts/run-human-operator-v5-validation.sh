#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
status="$repo_dir/results/human_operator_v5/development/gate_status.json"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["validation_unlocked"] else 3)' "$status" || {
  echo "V5 validation is prospectively locked because development gates failed; no validation job was started." >&2
  exit 3
}
echo "Gate file says unlocked, but no automatic command is provided without a frozen execution commit." >&2
exit 4
