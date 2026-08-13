#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export THERMO_CACHE="${THERMO_CACHE:-/workspace/.cache/thermoagent}"
mkdir -p "$HF_HOME" "$THERMO_CACHE" results/logs/setup results/reproducibility

if [[ ! -x .venv/bin/python ]]; then
  if [[ -e .venv ]]; then
    echo ".venv exists but is incomplete; move it aside before rerunning setup" >&2
    exit 2
  fi
  python3 -m venv --system-site-packages .venv
fi

# Standard pip sees inherited system-site distributions and therefore retains
# the image's validated torch 2.8.0+cu128 instead of resolving a second CUDA
# stack. Verify the invariant immediately after installation.
.venv/bin/python -m pip install --upgrade-strategy only-if-needed --requirement requirements-runpod.txt
.venv/bin/python -m pip install --no-deps --editable .
.venv/bin/python - <<'PY'
import torch
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
print("preserved torch runtime", torch.__version__, torch.version.cuda)
PY

.venv/bin/python - <<'PY'
import json
import platform
from importlib.metadata import version
from pathlib import Path
import torch

packages = [
    "thermoagent", "transformers", "accelerate", "bitsandbytes",
    "huggingface-hub", "safetensors", "pydantic", "pytest", "scipy",
    "pandas", "matplotlib", "networkx", "PyYAML",
    "scikit-learn",
]
record = {
    "python": platform.python_version(),
    "packages": {name: version(name) for name in packages},
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
Path("results/reproducibility/environment.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
PY

.venv/bin/pytest 2>&1 | tee results/logs/setup/tests-after-install.log
