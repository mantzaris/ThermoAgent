#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ -n "${THERMO_PYTHON:-}" ]]; then
  python_bin="$THERMO_PYTHON"
elif [[ -x "$repo_dir/.venv/bin/python" ]] && "$repo_dir/.venv/bin/python" -c 'import networkx, sklearn, torch' >/dev/null 2>&1; then
  python_bin="$repo_dir/.venv/bin/python"
else
  python_bin="$(command -v python3)"
fi
destination="$repo_dir/results/generalized_entropic_consensus_v6/reproducibility/pytest_v6.xml"
mkdir -p "$(dirname "$destination")"
"$python_bin" -m pytest -q \
  tests/test_v6_entropy.py \
  tests/test_v6_environment.py \
  tests/test_v6_policies.py \
  tests/test_v6_learnability.py \
  tests/test_v6_communication.py \
  tests/test_v6_qwen.py \
  tests/test_v6_replay.py \
  tests/test_v6_dashboard.py \
  tests/test_v6_artifacts.py \
  tests/test_v6_protocol.py \
  tests/test_v6_gates.py \
  tests/test_v6_training.py \
  --junitxml="$destination"
