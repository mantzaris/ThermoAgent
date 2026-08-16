from pathlib import Path

from thermoagent.v6_artifacts import build_index, verify_artifacts


def test_artifact_index_detects_mutation_and_crlf(tmp_path):
    root = tmp_path / "results"
    (root / "development").mkdir(parents=True)
    target = root / "development" / "value.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    build_index(root)
    assert verify_artifacts(root)["passed"]
    target.write_bytes(b"a,b\r\n1,3\r\n")
    report = verify_artifacts(root)
    assert not report["passed"]
    assert {value["reason"] for value in report["failure_details"]} == {
        "checksum_mismatch", "crlf_text",
    }
