#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$script_dir/runpod-exec.sh" python3 -c '
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
device = torch.device("cuda")
a = torch.randn((2048, 2048), device=device)
b = torch.randn((2048, 2048), device=device)
c = a @ b
torch.cuda.synchronize()
run_id = datetime.now(timezone.utc).strftime("setup-smoke-%Y%m%dT%H%M%SZ")
result = {
    "ok": bool(torch.isfinite(c).all().item()),
    "run_id": run_id,
    "host": platform.node(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "shape": list(c.shape),
}
artifact_dir = Path("runs") / run_id
artifact_dir.mkdir(parents=True, exist_ok=False)
(artifact_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
'
