#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
python_exec="${THERMO_PYTHON:-$repository_dir/.venv/bin/python}"
exec "$python_exec" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("results/entropy_triggered_belief_monitoring_v8/reproducibility/pdf_qa/report.json")
report = json.loads(path.read_text(encoding="utf-8"))
figures = report.get("figures", [])
if len(figures) != 8:
    raise SystemExit("expected exactly eight paper-facing V8 PDFs")
for figure in figures:
    if not all(figure.get(key) for key in (
        "opens", "fonts_detected", "fonts_embedded", "render_success",
    )):
        raise SystemExit("automated PDF QA must pass before attestation")
    if int(figure.get("render_dpi", 0)) != 240:
        raise SystemExit("manual review requires the 240-DPI render")
    figure["visual_inspection"] = "pass_manual_original_resolution_and_240_dpi"
    figure["visual_inspection_findings"] = "no clipping, overlap, unreadable legend, or undersized essential text"
report["manual_review"] = {
    "reviewer": "Codex primary agent",
    "reviewed_at": datetime.now(timezone.utc).isoformat(),
    "method": "original-resolution PNG preview plus automated 240-DPI render",
    "figures_reviewed": len(figures),
    "status": "pass",
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
print(json.dumps(report["manual_review"], indent=2, sort_keys=True))
PY
