import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _repository_files():
    roots = [
        ROOT / "thermoagent/statmech_llm_v14",
        ROOT / "configs/statmech_v14",
        ROOT / "tests/statmech_v14",
        ROOT / "results/collective_agent_statmech_v14",
        ROOT / "paper/jstat_v14",
    ]
    return [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]


def test_repository_package_excludes_raw_artifacts_weights_caches_and_oversized_files():
    files = _repository_files()
    forbidden = {".jsonl", ".safetensors", ".pt", ".bin", ".npy", ".npz", ".tar", ".zip", ".png"}
    assert not [path for path in files if path.suffix.lower() in forbidden]
    assert not [path for path in files if path.stat().st_size > 10 * 1024 * 1024]
    assert sum(path.stat().st_size for path in files) < 25 * 1024 * 1024


def test_no_secret_patterns_or_crlf_in_v14_text():
    patterns = (
        "jupyter" + "_token=",
        "api" + "_key=",
        "begin openssh" + " private key",
        "hf" + "_token=",
    )
    for path in _repository_files():
        if path.suffix.lower() in {".py", ".yaml", ".md", ".sh", ".tex", ".bib", ".csv", ".json"}:
            data = path.read_bytes()
            assert b"\r\n" not in data
            lowered = data.decode("utf-8", errors="ignore").lower()
            assert not any(pattern in lowered for pattern in patterns)


def test_manuscript_compiles_when_sources_are_present():
    paper = ROOT / "paper/jstat_v14/main.tex"
    if not paper.exists():
        pytest.skip("V14 manuscript not generated yet")
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    subprocess.run(command, cwd=paper.parent, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert (paper.parent / "main.pdf").exists()


def test_all_final_pdfs_have_embedded_fonts_and_extractable_text_when_present():
    pdfs = list((ROOT / "results/collective_agent_statmech_v14/figures/pdf").glob("*.pdf"))
    manuscript = ROOT / "paper/jstat_v14/main.pdf"
    if manuscript.exists():
        pdfs.append(manuscript)
    if not pdfs:
        pytest.skip("V14 PDFs not generated yet")
    for path in pdfs:
        fonts = subprocess.check_output(["pdffonts", str(path)], text=True)
        assert re.search(r"\byes\b", fonts.lower())
        text = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
        assert text.strip()
