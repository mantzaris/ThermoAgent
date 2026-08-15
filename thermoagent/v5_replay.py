"""Exact V5 event-ledger and metric regeneration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .events import EventLedger, sha256_file
from .v5_environment import payload_digest
from .v5_experiments import atomic_json


def replay_v5_episode(episode_path: Path) -> Dict[str, Any]:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    ledgers = list(episode_path.parent.glob("events.jsonl*"))
    if len(ledgers) != 1:
        raise ValueError("expected exactly one V5 event ledger")
    ledger_path = ledgers[0]
    ledger = EventLedger.read_jsonl(ledger_path)
    if ledger.digest() != episode["event_ledger_digest"]:
        raise ValueError("V5 ledger digest mismatch")
    tapes = [event for event in ledger.events if event.kind == "v5_stochastic_tape"]
    if len(tapes) != 1 or tapes[0].payload["digest"] != episode["stochastic_tape_digest"]:
        raise ValueError("V5 stochastic tape mismatch")
    branches = [event for event in ledger.events if event.kind == "counterfactual_branch"]
    for event in branches:
        if event.payload["rng_digest_with"] != event.payload["rng_digest_without"]:
            raise ValueError("V5 counterfactual RNG mismatch")
        if event.payload["rng_digest_with"] != episode["stochastic_tape_digest"]:
            raise ValueError("V5 counterfactual tape provenance mismatch")
        regenerated = float(event.payload["loss_without"]) - float(event.payload["loss_with"])
        if abs(regenerated - float(event.payload["causal_effect"])) > 1e-12:
            raise ValueError("V5 counterfactual metric mismatch")
    transitions = [event for event in ledger.events if event.kind == "v5_state_transition"]
    if len(transitions) != 1:
        raise ValueError("V5 episode must contain one panel transition")
    transition = transitions[0].payload
    if payload_digest(transition["before"]) != transition["before_digest"]:
        raise ValueError("V5 before-state digest mismatch")
    if payload_digest(transition["after"]) != transition["after_digest"]:
        raise ValueError("V5 after-state digest mismatch")
    if not transition["conservation"]["feasible"]:
        raise ValueError("V5 replay found infeasible resource state")
    regenerated_after = float(transition["before"]["loss"]) - float(transition["operator_effect"])
    if abs(regenerated_after - float(transition["after"]["loss"])) > 1e-12:
        raise ValueError("V5 transition loss mismatch")
    for key, initial in transition["before"]["resources"].items():
        remaining = float(transition["after"]["resources"][key])
        used = float(transition["after"]["used"][key])
        if remaining < -1e-12 or used < -1e-12 or abs(float(initial) - remaining - used) > 1e-12:
            raise ValueError("V5 conservation mismatch for %s" % key)
    audits = [event for event in ledger.events if event.kind == "v5_privacy_audit"]
    if not audits or any(
        event.payload.get("private_state_leak")
        or event.payload.get("future_state_leak")
        or event.payload.get("operator_global_state_leak")
        for event in audits
    ):
        raise ValueError("V5 privacy audit failed")
    summary = episode["summary"]
    if int(summary["candidate_count"]) != len(branches):
        raise ValueError("V5 candidate count mismatch")
    return {
        "run_id": summary["run_id"],
        "episode_path": str(episode_path),
        "ledger_path": str(ledger_path),
        "events": len(ledger.events),
        "counterfactual_branches": len(branches),
        "maximum_conservation_residual": float(transition["conservation"]["maximum_residual"]),
        "privacy_audits": len(audits),
        "mismatches": 0,
        "ledger_sha256": sha256_file(ledger_path),
        "status": "passed",
    }


def replay_v5_results(results_root: Path, stages: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    allowed = set(stages) if stages is not None else None
    paths = [
        path for path in (results_root / "raw").glob("*/*/episode.json")
        if allowed is None or path.parents[1].name in allowed
    ]
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for path in sorted(paths):
        try:
            records.append(replay_v5_episode(path))
        except Exception as error:
            failures.append({"episode_path": str(path), "error": "%s: %s" % (type(error).__name__, error)})
    report = {
        "episodes_replayed": len(records),
        "failures": len(failures),
        "mismatches": len(failures),
        "maximum_conservation_residual": max(
            [float(row["maximum_conservation_residual"]) for row in records] or [0.0]
        ),
        "records": records,
        "failure_records": failures,
    }
    atomic_json(results_root / "reproducibility" / "replay" / "v5_replay_report.json", report)
    return report
