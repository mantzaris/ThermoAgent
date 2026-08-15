"""Exact ledger and metric regeneration checks for v4 episodes."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .events import EventLedger, sha256_file


def payload_digest(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def replay_v4_episode(episode_path: Path) -> Dict[str, Any]:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    ledgers = sorted(episode_path.parent.glob("events.jsonl*"))
    if len(ledgers) != 1:
        raise FileNotFoundError("expected exactly one v4 event ledger beside episode")
    ledger_path = ledgers[0]
    ledger = EventLedger.read_jsonl(ledger_path)
    if ledger.digest() != episode["event_ledger_digest"]:
        raise ValueError("v4 event-ledger digest mismatch")
    topology = next((event for event in ledger.events if event.kind == "topology_snapshot"), None)
    if topology is None:
        raise ValueError("v4 ledger lacks topology snapshot")
    initial = topology.payload["initial_state"]
    if payload_digest(initial) != topology.payload["initial_state_digest"]:
        raise ValueError("v4 initial-state digest mismatch")
    transitions = [event for event in ledger.events if event.kind == "v4_state_transition"]
    if len(transitions) != len(episode["time_series"]):
        raise ValueError("v4 transition/time-series length mismatch")
    maximum_residual = 0.0
    reconstructed_losses: List[float] = []
    for index, event in enumerate(transitions):
        payload = event.payload
        if payload_digest(payload["before"]) != payload["before_digest"]:
            raise ValueError("v4 before-state digest mismatch at transition %d" % index)
        if payload_digest(payload["after"]) != payload["after_digest"]:
            raise ValueError("v4 after-state digest mismatch at transition %d" % index)
        conservation = payload["conservation"]
        maximum_residual = max(maximum_residual, float(conservation["maximum_residual"]))
        if not conservation["feasible"]:
            raise ValueError("v4 infeasible state in replay")
        if any(float(value) < -1e-12 for value in payload["after"]["resources"].values()):
            raise ValueError("v4 replay found negative inventory")
        reconstructed_losses.append(float(payload["loss"]))
        recorded = float(episode["time_series"][index]["loss"])
        if abs(recorded - reconstructed_losses[-1]) > 1e-12:
            raise ValueError("v4 replay metric mismatch at transition %d" % index)
    primary = float(sum(reconstructed_losses))
    if abs(primary - float(episode["metrics"]["primary_outcome"])) > 1e-10:
        raise ValueError("v4 primary outcome failed regeneration")
    privacy_audits = [event for event in ledger.events if event.kind == "information_boundary_audit"]
    for event in privacy_audits:
        if event.payload.get("private_state_leak") or event.payload.get("future_state_leak"):
            raise ValueError("v4 replay found operator-view privacy failure")
    counterfactuals = [event for event in ledger.events if event.kind == "counterfactual_branch"]
    if any(row.payload.get("rng_digest_with") != row.payload.get("rng_digest_without") for row in counterfactuals):
        raise ValueError("v4 counterfactual branches did not restore common randomness")
    return {
        "run_id": episode["run_id"],
        "episode_path": str(episode_path),
        "ledger_path": str(ledger_path),
        "event_count": len(ledger.events),
        "transition_count": len(transitions),
        "reconstructed_primary_outcome": primary,
        "maximum_conservation_residual": maximum_residual,
        "privacy_audits": len(privacy_audits),
        "counterfactual_branches": len(counterfactuals),
        "mismatches": 0,
        "ledger_sha256": sha256_file(ledger_path),
        "status": "passed",
    }


def replay_v4_results(results_root: Path, stages: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    raw = results_root / "raw"
    allowed = set(stages) if stages is not None else None
    paths = [
        path for path in raw.glob("*/*/episode.json")
        if allowed is None or path.parents[1].name in allowed
    ]
    records = [replay_v4_episode(path) for path in sorted(paths)]
    report = {
        "episodes_replayed": len(records),
        "mismatches": sum(int(row["mismatches"]) for row in records),
        "maximum_conservation_residual": max(
            (float(row["maximum_conservation_residual"]) for row in records), default=0.0
        ),
        "records": records,
    }
    destination = results_root / "reproducibility" / "v4_replay_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
