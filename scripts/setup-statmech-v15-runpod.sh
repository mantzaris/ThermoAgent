#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  if [[ -e .venv ]]; then
    echo ".venv exists but is incomplete; preserve and inspect it before recovery" >&2
    exit 2
  fi
  python3 -m venv --system-site-packages .venv
  .venv/bin/python -m pip install --upgrade-strategy only-if-needed --requirement requirements-runpod.txt
  .venv/bin/python -m pip install --no-deps --editable .
fi

.venv/bin/python -m pip install \
  --upgrade-strategy only-if-needed \
  --requirement configs/statmech_v15/requirements-runpod-v15.txt

.venv/bin/python - <<'PY'
from importlib.metadata import version
import torch

expected = {
    "transformers": "4.55.4",
    "bitsandbytes": "0.47.0",
    "sentencepiece": "0.2.0",
    "protobuf": "5.29.5",
}
observed = {name: version(name) for name in expected}
assert observed == expected, (observed, expected)
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable"
print("V15 RunPod environment ready")
print("torch", torch.__version__, "cuda", torch.version.cuda)
for name in sorted(observed):
    print(name, observed[name])
PY
