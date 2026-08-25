#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-/workspace/ThermoAgent-v15-model-cache/huggingface}"
mkdir -p "$HF_HOME"

PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
if [[ "$($PYTHON_BOOTSTRAP -c 'import platform; print(platform.python_version())')" != "3.12.3" ]]; then
  if command -v python3.12 >/dev/null 2>&1 \
    && [[ "$(python3.12 -c 'import platform; print(platform.python_version())')" == "3.12.3" ]]; then
    PYTHON_BOOTSTRAP=python3.12
  else
    echo "V15 reconstruction requires the original Python 3.12.3 runtime" >&2
    exit 2
  fi
fi

if [[ ! -x .venv/bin/python ]]; then
  if [[ -e .venv ]]; then
    echo ".venv exists but is incomplete; preserve and inspect it before recovery" >&2
    exit 2
  fi
  "$PYTHON_BOOTSTRAP" -m venv --system-site-packages .venv
fi

# Fresh RunPod images do not necessarily retain the CUDA-enabled system torch
# that the original V15 environment inherited.  Check the exact tested build
# before installing anything; if it is absent or different, install only the
# official CUDA 12.8 wheel into the venv.  This is idempotent on the previously
# tested base image and avoids accidentally accepting a CPU-only wheel.
if ! .venv/bin/python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(
    0
    if torch.__version__ == "2.8.0+cu128"
    and torch.version.cuda == "12.8"
    and torch.cuda.is_available()
    else 1
)
PY
then
  .venv/bin/python -m pip install \
    --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.8.0'
fi

.venv/bin/python -m pip install \
  --upgrade-strategy only-if-needed \
  --requirement requirements-runpod.txt

# Make the checkout importable without asking setuptools to write an
# ``*.egg-info`` directory into the repository.  Formal reconstruction runs
# from the clean frozen checkout, whose current working directory remains first
# on ``sys.path``; this path file is only the fallback for working-tree tools.
SITE_PACKAGES="$(.venv/bin/python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
printf '%s\n' "$ROOT" > "$SITE_PACKAGES/thermoagent-v15-reconstruction.pth"

.venv/bin/python -m pip install \
  --upgrade-strategy only-if-needed \
  --requirement configs/statmech_v15/requirements-runpod-v15.txt

.venv/bin/python - <<'PY'
from importlib.metadata import version
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch

expected = {
    "transformers": "4.55.4",
    "tokenizers": "0.21.4",
    "accelerate": "1.10.1",
    "bitsandbytes": "0.47.0",
    "huggingface-hub": "0.34.4",
    "safetensors": "0.6.2",
    "pydantic": "2.11.7",
    "pytest": "8.4.1",
    "numpy": "2.1.2",
    "scipy": "1.16.1",
    "pandas": "2.3.1",
    "matplotlib": "3.10.5",
    "networkx": "3.5",
    "PyYAML": "6.0.2",
    "PyMuPDF": "1.28.2",
    "scikit-learn": "1.5.2",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
    "sentencepiece": "0.2.0",
    "protobuf": "5.29.5",
}
observed = {name: version(name) for name in expected}
assert observed == expected, (observed, expected)
assert platform.python_version() == "3.12.3", platform.python_version()
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable"
print("V15 RunPod environment ready")
print("torch", torch.__version__, "cuda", torch.version.cuda)
for name in sorted(observed):
    print(name, observed[name])

artifact_root = Path(
    os.environ.get(
        "THERMO_V15_ARTIFACT_ROOT",
        "/workspace/ThermoAgent-v15-reconstruction-b309f0ab",
    )
).resolve()
repository = Path.cwd().resolve()
if artifact_root == repository or repository in artifact_root.parents:
    raise RuntimeError("environment manifest must remain outside the repository")
protocol_path = repository / "configs/statmech_v15/protocol_frozen.yaml"
protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
repository_commit = subprocess.check_output(
    ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
).strip()
identity = {
    "reconstruction_label": "fresh-v15-b309f0ab",
    "repository_commit": repository_commit,
    "protocol_sha256": protocol_sha256,
    "qwen_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    "granite_revision": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
}
identity_path = artifact_root / "reproducibility/reconstruction_identity.json"
if artifact_root.exists() and any(artifact_root.iterdir()):
    if not identity_path.is_file():
        raise RuntimeError(
            "nonempty artifact root lacks a reconstruction identity; preserve and inspect it"
        )
    existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if any(existing_identity.get(key) != value for key, value in identity.items()):
        raise RuntimeError("artifact root belongs to a different reconstruction")
identity_path.parent.mkdir(parents=True, exist_ok=True)
identity_temporary = identity_path.with_suffix(".tmp")
identity_temporary.write_text(
    json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
identity_temporary.replace(identity_path)
destination = artifact_root / "reproducibility/environment_reconstruction.json"
destination.parent.mkdir(parents=True, exist_ok=True)
disk = shutil.disk_usage(artifact_root.parent)
repository_disk = shutil.disk_usage(repository)
gpu_query = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip().splitlines()
nvcc_path = shutil.which("nvcc")
nvcc_version = (
    subprocess.check_output([nvcc_path, "--version"], text=True).strip()
    if nvcc_path
    else None
)
page_size = int(os.sysconf("SC_PAGE_SIZE"))
physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
payload = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "packages": {**observed, "torch": torch.__version__},
    "torch_cuda": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_count": int(torch.cuda.device_count()),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "nvidia_smi_gpu_status_rows": gpu_query,
    "nvcc_version_output": nvcc_version,
    "physical_ram_bytes": page_size * physical_pages,
    "artifact_root": str(artifact_root),
    "reconstruction_identity": identity,
    "huggingface_cache_root": str(Path(os.environ["HF_HOME"]).resolve()),
    "disk_total_bytes": int(disk.total),
    "disk_free_bytes": int(disk.free),
    "repository_filesystem_total_bytes": int(repository_disk.total),
    "repository_filesystem_free_bytes": int(repository_disk.free),
}
temporary = destination.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(destination)
print("environment manifest", destination)
PY
