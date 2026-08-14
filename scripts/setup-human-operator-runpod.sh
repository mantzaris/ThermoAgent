#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export THERMO_CACHE="${THERMO_CACHE:-/workspace/.cache/thermoagent}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/thermoagent-v3-pycache}"
v3_root="$repo_dir/results/human_operator_v3"
mkdir -p "$HF_HOME" "$THERMO_CACHE" "$v3_root/logs/setup" "$v3_root/reproducibility"

[[ -x .venv/bin/python ]] || {
  echo "The compatible isolated .venv is missing; use setup-runpod.sh only after inspecting the Pod." >&2
  exit 2
}

if ! .venv/bin/python - <<'PY'
import accelerate, bitsandbytes, matplotlib, numpy, pandas, sklearn, torch, transformers
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert transformers.__version__ == "4.55.4", transformers.__version__
PY
then
  echo "A required package is absent or incompatible; installing only the pinned project requirements." >&2
  .venv/bin/python -m pip install --upgrade-strategy only-if-needed --requirement requirements-runpod.txt
  .venv/bin/python -m pip install --no-deps --editable .
fi

.venv/bin/python - <<'PY'
import json, platform
from importlib.metadata import version
from pathlib import Path
import torch
packages = ["thermoagent", "transformers", "accelerate", "bitsandbytes", "numpy", "pandas", "scikit-learn", "matplotlib", "pytest"]
record = {
    "python": platform.python_version(),
    "packages": {name: version(name) for name in packages},
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0),
    "evidence_boundary": "non-secret v3 RunPod environment audit",
}
path = Path("results/human_operator_v3/reproducibility/runpod_environment.json")
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True))
PY
nvidia-smi > "$v3_root/logs/setup/nvidia-smi.txt"
.venv/bin/python -m pytest -q 2>&1 | tee "$v3_root/logs/setup/remote_complete_test_suite.log"
