import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

from thermoagent.v8_experiments import run_v8_episode
from thermoagent.v8_replay import _ledger_audit, _packed_rows, replay_v8_results
from thermoagent.v8_trigger import TriggerConfig


def test_dynamic_delta_ledger_replays_exactly(tmp_path: Path):
    output = run_v8_episode(
        application="utility_restoration", complexity="small",
        coupling="low", fragmentation="medium", network_disruption="medium",
        topology_family="grid", environment_seed=889802,
        trigger_config=TriggerConfig(
            method="generalized_information", tau_on=0.125, tau_off=0.04,
            maximum_silence_steps=30,
        ),
        encoding="uint8_simplex", results_root=tmp_path,
        stage="test_formal", ledger_scope="dynamic_delta",
    )
    report = replay_v8_results(tmp_path)
    assert output["stored_event_count"] < output["summary"]["event_count"]
    assert report["episodes_replayed"] == 1
    assert report["replay_mismatches"] == 0


def test_replay_accepts_versioned_metric_missing_from_both_records():
    summary = {
        "run_id": "old-pilot", "stage": "pilots", "application": "humanitarian",
        "service_loss": 1.0, "net_causal_utility": 0.5,
        "fully_counted_messages": 3, "fully_counted_bytes": 20,
        "sketch_on_wire_bytes": 12, "maximum_conservation_residual": 0.0,
    }
    event = {
        "event_id": "E00000001", "step": 1, "kind": "metric",
        "actor": "evaluator", "payload": dict(summary), "private_to": "evaluator",
    }
    canonical = json.dumps(event, sort_keys=True)
    payload = (canonical + "\n").encode("utf-8")
    episode = {
        "summary": summary,
        "event_ledger_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "event_ledger_sha256": "stored-versioned-ledger",
    }
    row = _ledger_audit(
        payload, episode, stored_sha256="stored-versioned-ledger",
        logical_sha256=hashlib.sha256(payload).hexdigest(), locator="memory",
    )
    assert row["status"] == "pass"


def test_replay_rejects_one_sided_versioned_metric_omission():
    summary = {
        "run_id": "bad-pilot", "stage": "pilots", "application": "humanitarian",
        "service_loss": 1.0, "net_causal_utility": 0.5,
        "fully_counted_messages": 3, "fully_counted_bytes": 20,
        "sketch_on_wire_bytes": 12, "maximum_conservation_residual": 0.0,
        "primary_distributed_state_error": 0.1,
    }
    metric = dict(summary)
    metric.pop("primary_distributed_state_error")
    event = {
        "event_id": "E00000001", "step": 1, "kind": "metric",
        "actor": "evaluator", "payload": metric, "private_to": "evaluator",
    }
    canonical = json.dumps(event, sort_keys=True)
    payload = (canonical + "\n").encode("utf-8")
    episode = {
        "summary": summary,
        "event_ledger_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "event_ledger_sha256": "stored-one-sided",
    }
    row = _ledger_audit(
        payload, episode, stored_sha256="stored-one-sided",
        logical_sha256=hashlib.sha256(payload).hexdigest(), locator="memory",
    )
    assert row["status"] == "mismatch"
    assert "metric_regeneration" in row["mismatch_reasons"]


def test_packed_replay_preserves_but_does_not_promote_partial_run(tmp_path: Path):
    summary = {
        "run_id": "complete", "stage": "pilot", "application": "humanitarian",
        "service_loss": 1.0, "net_causal_utility": 0.5,
        "fully_counted_messages": 3, "fully_counted_bytes": 20,
        "sketch_on_wire_bytes": 12, "maximum_conservation_residual": 0.0,
    }
    event = {
        "event_id": "E00000001", "step": 1, "kind": "metric",
        "actor": "evaluator", "payload": dict(summary), "private_to": "evaluator",
    }
    canonical = json.dumps(event, sort_keys=True)
    ledger = (canonical + "\n").encode("utf-8")
    stored_hash = "stored-complete-ledger"
    episode = json.dumps({
        "summary": summary,
        "event_ledger_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "event_ledger_sha256": stored_hash,
    }, sort_keys=True).encode("utf-8")
    partial = ledger
    archive_path = tmp_path / "packed.tar.xz"
    members = {
        "partial/events.jsonl": partial,
        # Deliberately put the complete ledger before its episode payload.
        "complete/events.jsonl": ledger,
        "complete/episode.json": episode,
    }
    with tarfile.open(archive_path, mode="w:xz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    manifest_path = archive_path.with_suffix(".manifest.csv")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "archive_member", "stored_sha256", "logical_sha256",
        ], lineterminator="\n")
        writer.writeheader()
        for name, payload in members.items():
            writer.writerow({
                "archive_member": name,
                "stored_sha256": stored_hash if name == "complete/events.jsonl" else "unused",
                "logical_sha256": hashlib.sha256(payload).hexdigest(),
            })

    rows = _packed_rows(archive_path)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "complete"
    assert rows[0]["status"] == "pass"
