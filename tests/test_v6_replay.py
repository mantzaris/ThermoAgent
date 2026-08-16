import gzip
import json
from pathlib import Path

from thermoagent.events import EventLedger, sha256_file
from thermoagent.v6_experiments import read_episode_json, run_episode
from thermoagent.v6_replay import replay_all, replay_episode


def test_v6_ledger_replay_regenerates_metrics_conservation_and_privacy(tmp_path):
    root = tmp_path / "results"
    run_episode(
        Path.cwd(), root, "test_replay", "humanitarian", "compound",
        "private_fragmented", 66920, "action_value_margin", 0.5,
        "event_triggered", resume=False,
    )
    episode = next((root / "raw" / "test_replay").glob("*/episode.json*"))
    result = replay_episode(root, episode)
    assert result["status"] == "pass"
    assert result["metric_regeneration_match"]
    assert result["privacy_boundary_pass"]
    assert result["maximum_reconstructed_conservation_residual"] <= 1e-9


def test_replay_separates_retained_invalid_pilot_from_formal_gate(tmp_path):
    root = tmp_path / "results"
    for stage in ("development_dynamic", "pilot_v1"):
        run_episode(
            Path.cwd(), root, stage, "humanitarian", "compound",
            "private_fragmented", 66921, "action_value_margin", 0.5,
            "event_triggered", resume=False,
        )

    pilot_episode = next((root / "raw" / "pilot_v1").glob("*/episode.json*"))
    payload = read_episode_json(pilot_episode)
    ledger_path = root / payload["summary"]["event_ledger_path"]
    original = EventLedger.read_jsonl(ledger_path)
    broken = EventLedger()
    changed = False
    for event in original.events:
        private_to = event.private_to
        if event.kind == "v6_consensus_state" and not changed:
            private_to = None
            changed = True
        broken.append(
            event.step, event.kind, event.actor, event.payload,
            private_to=private_to,
        )
    assert changed
    broken.write_jsonl(ledger_path)
    payload["summary"]["event_ledger_sha256"] = sha256_file(ledger_path)
    payload["summary"]["event_ledger_digest"] = broken.digest()
    with pilot_episode.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(
                (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
            )

    report = replay_all(root)
    assert report["episodes_replayed"] == 2
    assert report["replay_mismatches"] == 1
    assert report["formal_episodes_replayed"] == 1
    assert report["formal_replay_mismatches"] == 0
    assert report["retained_pilot_mismatches"] == 1
    assert report["formal_status"] == "pass"
