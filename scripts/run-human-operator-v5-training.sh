#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/results/human_operator_v5/logs"
cd "$repo_dir"
python3 -c '
from pathlib import Path
from thermoagent.v5_training import train_multiseed
root = Path(".").resolve()
print(train_multiseed(root, root / "results" / "human_operator_v5"))
' 2>&1 | tee "$repo_dir/results/human_operator_v5/logs/multiseed_training.log"
