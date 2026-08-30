from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_repository_facing_v13_excludes_raw_prompts_weights_caches_and_large_files():
    roots = [
        ROOT / "thermoagent/statmech_llm/replication",
        ROOT / "configs/statmech_llm/replication",
        ROOT / "tests/statmech_llm/replication",
        ROOT / "results/JSTAT/stages/replication",
        ROOT / "paper/JSTAT",
    ]
    files = [
        path for root in roots if root.exists() for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    forbidden = {".jsonl", ".safetensors", ".pt", ".bin", ".npy", ".npz", ".tar", ".zip"}
    assert not [path for path in files if path.suffix.lower() in forbidden]
    assert not [path for path in files if path.stat().st_size > 10 * 1024 * 1024]
    assert sum(path.stat().st_size for path in files) < 30 * 1024 * 1024


def test_no_obvious_secret_material_in_v13_text():
    roots = [ROOT / "thermoagent/statmech_llm/replication", ROOT / "configs/statmech_llm/replication", ROOT / "tests/statmech_llm/replication"]
    prohibited = (
        "jupyter" + "_token=",
        "api" + "_key=",
        "BEGIN OPENSSH" + " PRIVATE KEY",
        "hf" + "_token=",
    )
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".md", ".sh", ".tex"}:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
                assert not any(token.lower() in text for token in prohibited)
