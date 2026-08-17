from pathlib import Path

from thermoagent.v8_index import build_v8_index, verify_v8_index


def test_v8_artifact_index_detects_content_changes(tmp_path: Path):
    (tmp_path / "tables").mkdir()
    artifact = tmp_path / "tables" / "example.csv"
    with artifact.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("a,b\n1,2\n")
    report = build_v8_index(tmp_path)
    assert report["status"] == "pass"
    assert verify_v8_index(tmp_path)["status"] == "pass"
    with artifact.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("a,b\n1,3\n")
    mismatch = verify_v8_index(tmp_path)
    assert mismatch["status"] == "fail"
    assert mismatch["mismatches"] == ["tables/example.csv:sha256"]


def test_v8_artifact_index_rejects_crlf(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_bytes(b"a,b\r\n1,2\r\n")
    report = build_v8_index(tmp_path)
    assert report["status"] == "fail"
    assert report["crlf_text_files"] == ["bad.csv"]
