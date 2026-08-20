from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_repository_facing_v13_excludes_raw_prompts_weights_caches_and_large_files():
    roots = [
        ROOT / "thermoagent/statmech_llm_v13",
        ROOT / "configs/statmech_v13",
        ROOT / "tests/statmech_v13",
        ROOT / "results/collective_agent_statmech_v13",
        ROOT / "paper/jstat_v13",
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
    roots = [ROOT / "thermoagent/statmech_llm_v13", ROOT / "configs/statmech_v13", ROOT / "tests/statmech_v13"]
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
