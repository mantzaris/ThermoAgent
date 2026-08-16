from thermoagent.v7_artifacts import build_index, crlf_audit, verify_index


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
