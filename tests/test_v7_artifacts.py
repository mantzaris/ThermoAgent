from thermoagent.v5_experiments import atomic_json, write_csv
from thermoagent.v7_artifacts import build_index, crlf_audit, verify_index
from thermoagent.v7_compaction import compact_v7, verify_compaction
from thermoagent.v7_io import read_csv_artifact, read_json_artifact


def test_v7_artifact_index_detects_checksum_changes(tmp_path):
    repository = tmp_path / "repository"
    results = repository / "results" / "complexity_entropic_coordination_v7"
    (results / "development").mkdir(parents=True)
    artifact = results / "development" / "value.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")
    summary = build_index(results)
    assert summary["indexed_artifacts_excluding_index"] == 1
    assert verify_index(results)["pass"]
    artifact.write_text("a,b\n1,3\n", encoding="utf-8")
    assert not verify_index(results)["pass"]


def test_v7_crlf_audit_fails_for_generated_text(tmp_path):
    repository = tmp_path / "repository"
    results = repository / "results" / "complexity_entropic_coordination_v7"
    results.mkdir(parents=True)
    value = results / "bad.csv"
    value.write_bytes(b"a,b\r\n1,2\r\n")
    report = crlf_audit(repository, results)
    assert not report["pass"]
    assert report["crlf_files"]


def test_v7_lossless_compaction_preserves_candidates_and_is_idempotent(tmp_path):
    results = tmp_path / "results" / "complexity_entropic_coordination_v7"
    run = results / "raw" / "pilot" / "run-1"
    candidates = [{"agent_id": "a-1", "harmful": False, "utility": 0.125}]
    episode = {"summary": {"run_id": "run-1"}, "candidates": candidates}
    atomic_json(run / "episode.json", episode)
    write_csv(run / "candidate_decisions.csv", candidates)
    write_csv(results / "pilot" / "candidate_decisions.csv", candidates)

    report = compact_v7(results)
    assert report["status"] == "pass"
    assert report["canonical_episode_artifacts"] == 1
    assert not (run / "episode.json").exists()
    assert not (run / "candidate_decisions.csv").exists()
    assert read_json_artifact(run / "episode.json") == episode
    assert len(read_csv_artifact(results / "pilot" / "candidate_decisions.csv")) == 1
    assert verify_compaction(results)["status"] == "pass"
    assert compact_v7(results)["status"] == "pass"
