import gzip
import json
import lzma
from pathlib import Path

from thermoagent.v5_experiments import atomic_json
from thermoagent.v8_compaction import pack_completed_stage, pack_retained_incomplete_stage


def test_v8_stage_compaction_is_lossless_and_narrowly_scoped(tmp_path):
    root = tmp_path / "v8"
    stage = "pilots"
    expected = {}
    for application in ("humanitarian", "utility_restoration"):
        run = root / "raw" / stage / ("v8-pilots-%s-small-1" % application)
        run.mkdir(parents=True)
        (run / "episode.json").write_text('{"ok":true}\n', encoding="utf-8")
        with (run / "table.csv.gz").open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
                handle.write(b"a,b\n1,2\n")
        with lzma.open(run / "events.jsonl.xz", "wb") as handle:
            handle.write(b'{"event":1}\n')
        expected[application] = {
            "episode.json": b'{"ok":true}\n',
            "table.csv": b"a,b\n1,2\n",
            "events.jsonl": b'{"event":1}\n',
        }
    atomic_json(root / stage / "execution_summary.json", {
        "completed_episodes": 2,
    })
    report = pack_completed_stage(root, stage)
    assert report["status"] == "pass"
    assert not (root / "raw" / stage).exists()
    import tarfile
    for application in expected:
        archive_path = root / "raw" / "packed" / stage / (application + "-part01.tar.xz")
        with tarfile.open(archive_path, "r:xz") as archive:
            members = {
                Path(member.name).name: archive.extractfile(member).read()
                for member in archive.getmembers()
            }
        assert members == expected[application]


def test_v8_retained_incomplete_compaction_preserves_partial_metadata(tmp_path):
    root = tmp_path / "v8"
    stage = "invalidated_stage"
    complete = root / "raw" / stage / "v8-x-humanitarian-complete"
    partial = root / "raw" / stage / "v8-x-humanitarian-partial"
    complete.mkdir(parents=True)
    partial.mkdir(parents=True)
    (complete / "episode.json").write_text('{"summary":{"run_id":"r"}}\n')
    with lzma.open(complete / "events.jsonl.xz", "wb") as handle:
        handle.write(b'{"event":1}\n')
    (partial / "events.jsonl.xz.tmp").write_bytes(b"partial")
    report = pack_retained_incomplete_stage(
        root, stage, expected_complete=1, expected_partial=1,
    )
    assert report["complete_episodes"] == 1
    assert report["partial_run_directories"] == 1
    assert not report["eligible_as_scientific_evidence"]
    assert not (root / "raw" / stage).exists()
