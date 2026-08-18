"""Automated PDF validation with external-only 300-DPI renders."""

from __future__ import annotations

import subprocess
import re
from pathlib import Path
from typing import Dict, List

from .workflow import artifact_root, atomic_json, sha256_file, utc_now


def validate_pdfs(repository: Path) -> Dict[str, object]:
    repository = Path(repository)
    pdf_root = repository / "results/llm_agent_entropy_v11/figures/pdf"
    render_root = artifact_root() / "pdf_qa/renders_300dpi"
    render_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    paths = list(sorted(pdf_root.glob("*.pdf")))
    manuscript = repository / "paper/jstat_v11/main.pdf"
    if manuscript.exists():
        paths.append(manuscript)
    for path in paths:
        info = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True)
        page_line = next(line for line in info.stdout.splitlines() if line.startswith("Pages:"))
        page_count = int(page_line.split(":", 1)[1].strip())
        extracted_process = subprocess.run(
            ["pdftotext", str(path), "-"], check=True, capture_output=True, text=True
        )
        extracted = extracted_process.stdout
        render_stem = path.stem if path.parent == pdf_root else "paper_jstat_v11_" + path.stem
        render_prefix = render_root / render_stem
        render_process = subprocess.run(
            ["pdftoppm", "-png", "-r", "300", str(path), str(render_prefix)],
            check=False,
            capture_output=True,
            text=True,
        )
        render_paths = [str(item) for item in sorted(render_root.glob(render_stem + "-*.png"))]
        render_success = render_process.returncode == 0 and len(render_paths) == page_count
        fonts_embedded = False
        font_output = ""
        try:
            process = subprocess.run(["pdffonts", str(path)], check=True, capture_output=True, text=True)
            font_output = process.stdout
            font_rows = [line for line in process.stdout.splitlines()[2:] if line.strip()]
            font_flags = [
                re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", line)
                for line in font_rows
            ]
            fonts_embedded = bool(font_rows) and all(
                match is not None and match.group(1).lower() == "yes" for match in font_flags
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            fonts_embedded = False
        rows.append(
            {
                "path": path.relative_to(repository).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "pages": page_count,
                "opens": True,
                "render_300dpi_success": render_success,
                "render_paths_external": render_paths,
                "text_extractable": len(extracted.strip()) > 20,
                "fonts_embedded": fonts_embedded,
                "font_check_excerpt": font_output[:1000],
                "manual_original_resolution_inspection": False,
                "manual_300dpi_inspection": False,
                "clipping_or_overlap": "not_yet_inspected",
            }
        )
    report = {
        "generated_at": utc_now(),
        "pdf_count": len(rows),
        "all_open": bool(rows) and all(row["opens"] for row in rows),
        "all_render": bool(rows) and all(row["render_300dpi_success"] for row in rows),
        "all_text_extractable": bool(rows) and all(row["text_extractable"] for row in rows),
        "all_fonts_embedded": bool(rows) and all(row["fonts_embedded"] for row in rows),
        "files": rows,
    }
    atomic_json(report, repository / "results/llm_agent_entropy_v11/reproducibility/pdf_qa.json")
    return report
