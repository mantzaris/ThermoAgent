#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${THERMO_PYTHON:-$repo_dir/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"

stage_name="${1:-pilot_v5_reference}"
seed_start="${2:-60941}"

"$python_bin" - "$stage_name" "$seed_start" <<'PY'
from pathlib import Path
import sys
from thermoagent.v6_experiments import run_matrix

stage_name = sys.argv[1]
seed_start = int(sys.argv[2])
root = Path("results/generalized_entropic_consensus_v6")
run_matrix(
    Path.cwd(), root, stage_name,
    ("commercial", "humanitarian", "utility_restoration"),
    ("telemetry_integrity", "partition", "compound"),
    ("private_fragmented", "public_shared"),
    tuple(range(seed_start, seed_start + 5)),
    ("never_act",),
    (0.5,), ("event_triggered",), escalation_slots=1,
)
PY

echo "V6 ${stage_name} complete"
