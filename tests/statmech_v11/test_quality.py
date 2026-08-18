from pathlib import Path
import re
import shutil
import subprocess

import pytest

from thermoagent.statmech_llm_v11.figures import generate_figures
from thermoagent.statmech_llm_v11.pdf_qa import validate_pdfs


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v11_gitignore_excludes_raw_and_model_artifacts():
    text = (repository_root() / ".gitignore").read_text(encoding="utf-8")
    for required in (
        "results/llm_agent_entropy_v11/raw/",
        "results/llm_agent_entropy_v11/transcripts/",
        "results/llm_agent_entropy_v11/**/*.safetensors",
        "results/llm_agent_entropy_v11/**/*.pt",
        "paper/jstat_v11/*.aux",
    ):
        assert required in text


def test_protocol_figures_have_source_data_and_pass_automated_pdf_qa(tmp_path, monkeypatch):
    external = tmp_path / "external"
    monkeypatch.setenv("THERMO_V11_ARTIFACT_ROOT", str(external))
    generated = generate_figures(tmp_path)
    assert len(generated) == 2
    assert (tmp_path / "results/llm_agent_entropy_v11/figures/source_data/figure_01_architecture.csv").exists()
    report = validate_pdfs(tmp_path)
    assert report["all_open"]
    assert report["all_render"]
    assert report["all_text_extractable"]
    assert report["all_fonts_embedded"]


def test_repository_facing_v11_paths_obey_size_limits():
    root = repository_root()
    paths = [
        root / "configs/statmech_v11",
        root / "thermoagent/statmech_llm_v11",
        root / "tests/statmech_v11",
        root / "results/llm_agent_entropy_v11",
        root / "paper/jstat_v11",
    ]
    files = [item for path in paths if path.exists() for item in path.rglob("*") if item.is_file()]
    assert all(item.stat().st_size < 10 * 1024 * 1024 for item in files)
    assert sum(item.stat().st_size for item in files) < 25 * 1024 * 1024


def test_jstat_manuscript_compiles_with_embedded_fonts_and_extractable_text():
    root = repository_root()
    for executable in ("latexmk", "pdfinfo", "pdffonts", "pdftotext"):
        assert shutil.which(executable), executable + " is required for manuscript QA"
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-cd", "paper/jstat_v11/main.tex"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    )
    manuscript = root / "paper/jstat_v11/main.pdf"
    info = subprocess.run(["pdfinfo", str(manuscript)], check=True, capture_output=True, text=True).stdout
    assert "Pages:" in info
    extracted = subprocess.run(
        ["pdftotext", str(manuscript), "-"], check=True, capture_output=True, text=True
    ).stdout
    assert "formal experiment did not run" in extracted
    font_output = subprocess.run(
        ["pdffonts", str(manuscript)], check=True, capture_output=True, text=True
    ).stdout
    font_rows = [line for line in font_output.splitlines()[2:] if line.strip()]
    flags = [re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", line) for line in font_rows]
    assert font_rows and all(match is not None and match.group(1) == "yes" for match in flags)


def test_formal_stage_is_locked_without_passing_qualification(tmp_path, monkeypatch):
    from thermoagent.statmech_llm_v11.formal import run_formal_network
    from thermoagent.statmech_llm_v11.workflow import atomic_json

    external = tmp_path / "external"
    monkeypatch.setenv("THERMO_V11_ARTIFACT_ROOT", str(external))
    monkeypatch.setenv("THERMO_V11_ENABLE_QWEN", "1")
    atomic_json({"formal_network_unlocked": False}, external / "qualification/analysis.json")
    with pytest.raises(RuntimeError, match="did not unlock"):
        run_formal_network(repository_root())
