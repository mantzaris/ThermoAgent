"""Exact V7 ledger, metric, privacy, and independent conservation replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .events import EventLedger, sha256_file
from .v5_experiments import atomic_json, write_csv


def replay_episode(results_root: Path, episode_path: Path) -> Dict[str, Any]:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    ledger_path = results_root / episode["event_ledger_path"]
    ledger = EventLedger.read_jsonl(ledger_path)
    sha_match = sha256_file(ledger_path) == episode["event_ledger_sha256"]
    digest_match = ledger.digest() == episode["event_ledger_digest"]
    metrics = [event.payload for event in ledger.events if event.kind == "metric"]
    summary = dict(episode["summary"])
    metric_match = len(metrics) == 1
    if metrics:
        for key in (
            "service_loss", "harmful_actions", "beneficial_actions",
            "physical_actions", "net_causal_utility", "total_messages",
            "total_bytes", "maximum_conservation_residual",
        ):
            metric_match = metric_match and bool(np.isclose(
                float(metrics[-1][key]), float(summary[key]), atol=1e-12,
            ))
    latest_resources: Dict[str, Mapping[str, Any]] = {}
    for event in ledger.events:
        if event.kind == "v7_resource_transition":
            latest_resources[str(event.payload["resource"])] = event.payload
    reconstructed: Dict[str, float] = {}
    for resource, values in latest_resources.items():
        reconstructed[resource] = float(
            float(values["initial"]) - float(values["remaining"])
            - float(values["consumed"]) - float(values["in_transit"])
            - float(values["delivered"]) - float(values["losses"])
        )
    maximum_residual = max([abs(value) for value in reconstructed.values()] or [0.0])
    privacy_failures = 0
    for event in ledger.events:
        if event.kind in ("v7_private_observation", "v7_belief_update", "v7_distributed_state"):
            privacy_failures += int(event.private_to is None)
        if event.kind == "v7_counterfactual_branch":
            privacy_failures += int(event.private_to != "evaluator")
    finite = all(
        np.isfinite(float(value))
        for value in summary.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    reasons = []
    if not sha_match:
        reasons.append("ledger_sha256")
    if not digest_match:
        reasons.append("ledger_digest")
    if not metric_match:
        reasons.append("metric_regeneration")
    if maximum_residual > 1e-9:
        reasons.append("conservation")
    if privacy_failures:
        reasons.append("privacy")
    if not finite:
        reasons.append("nonfinite")
    return {
        "episode_path": str(episode_path.relative_to(results_root)),
        "ledger_path": str(ledger_path.relative_to(results_root)),
        "run_id": summary["run_id"],
        "events": len(ledger.events),
        "ledger_sha256_match": sha_match,
        "ledger_digest_match": digest_match,
        "metric_regeneration_match": metric_match,
        "privacy_failures": privacy_failures,
        "maximum_reconstructed_conservation_residual": maximum_residual,
        "finite_metrics": finite,
        "status": "pass" if not reasons else "mismatch",
        "mismatch_reasons": ";".join(reasons),
    }


def replay_all(results_root: Path) -> Dict[str, Any]:
    paths = sorted((results_root / "raw").glob("**/episode.json"))
    rows = [replay_episode(results_root, path) for path in paths]
    destination = results_root / "reproducibility" / "replay"
    write_csv(destination / "ledger_replay_audit.csv", rows)
    report = {
        "episodes_replayed": len(rows),
        "replay_mismatches": sum(value["status"] != "pass" for value in rows),
        "maximum_conservation_residual": max(
            [value["maximum_reconstructed_conservation_residual"] for value in rows]
            or [0.0]
        ),
        "privacy_failures": sum(value["privacy_failures"] for value in rows),
        "status": "pass" if rows and all(value["status"] == "pass" for value in rows) else "fail",
    }
    atomic_json(destination / "replay_summary.json", report)
    return report
