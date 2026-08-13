#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export THERMO_CACHE="${THERMO_CACHE:-/workspace/.cache/thermoagent}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/thermoagent-pycache}"
v2_root="results/entropy_triggered_v2"
mkdir -p "$HF_HOME" "$THERMO_CACHE" \
  "$v2_root/logs/setup" "$v2_root/reproducibility"

if [[ ! -x .venv/bin/python ]]; then
  if [[ -e .venv ]]; then
    echo ".venv exists but is incomplete; preserve and diagnose it before setup" >&2
    exit 2
  fi
  python3 -m venv --system-site-packages .venv
fi

.venv/bin/python -m pip install --upgrade-strategy only-if-needed \
  --requirement requirements-runpod.txt
.venv/bin/python -m pip install --no-deps --editable .
.venv/bin/python - <<'PY'
import torch
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
print("preserved torch runtime", torch.__version__, torch.version.cuda)
PY

.venv/bin/python -m thermoagent capture-env \
  --results "$v2_root"
.venv/bin/python -m pytest -q 2>&1 \
  | tee "$v2_root/logs/setup/tests-after-install.log"
