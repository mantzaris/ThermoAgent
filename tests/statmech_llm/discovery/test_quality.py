from pathlib import Path

from thermoagent.statmech_llm.discovery.workflow import execution_source_checksum, sha256_file
from thermoagent.statmech_llm.workflow import tree_digest


ROOT = Path(__file__).resolve().parents[3]


def test_execution_source_checksum_is_deterministic_and_excludes_frozen_output():
    first = execution_source_checksum(ROOT)
    second = execution_source_checksum(ROOT)
    assert len(first) == 64 and first == second


def test_external_tree_summary_is_compact_and_content_sensitive(tmp_path):
    stage = tmp_path / "raw"
    stage.mkdir()
    (stage / "a.txt").write_text("alpha", encoding="utf-8")
    first = tree_digest(tmp_path)
    assert first["file_count"] == 1
    (stage / "a.txt").write_text("beta", encoding="utf-8")
    second = tree_digest(tmp_path)
    assert first["tree_sha256"] != second["tree_sha256"]


def test_no_repository_facing_v12_raw_or_model_artifacts():
    roots = [
        ROOT / "thermoagent/statmech_llm/discovery",
        ROOT / "configs/statmech_llm/discovery",
        ROOT / "tests/statmech_llm/discovery",
        ROOT / "results/JSTAT/stages/discovery",
        ROOT / "paper/JSTAT",
    ]
    forbidden = {".jsonl", ".safetensors", ".pt", ".bin", ".npy", ".npz", ".tar", ".zip"}
    files = [path for root in roots if root.exists() for path in root.rglob("*") if path.is_file()]
    assert not [path for path in files if path.suffix.lower() in forbidden]
    assert not [path for path in files if path.stat().st_size > 10 * 1024 * 1024]
    assert sum(path.stat().st_size for path in files) < 30 * 1024 * 1024
