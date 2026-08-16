"""Exact ledger and metric-regeneration audit for all V6 episodes."""

from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .events import EventLedger, sha256_file
from .v5_experiments import atomic_json, write_csv
from .v6_experiments import read_episode_json


PROHIBITED_DEPLOYABLE_KEYS = {
    "true_mode", "correct_action", "stochastic_tape", "future_outcome",
    "counterfactual_effect", "evaluator_distributed_error",
}


def _evidence_category(results_root: Path, episode_path: Path) -> str:
    """Separate frozen/formal evidence from retained design iterations.

    Early pilots are intentionally retained even when they document a privacy
    defect repaired before formal development. They must be replayed and
    counted, but are not eligible to pass or fail the frozen formal gate. This
    classification uses only the immutable namespace, never replay outcomes.
    """
    relative = episode_path.relative_to(results_root)
    raw_stage = relative.parts[1] if len(relative.parts) > 1 else "unknown"
    if raw_stage.startswith("development_") or raw_stage == "qwen":
        return "formal_or_qualified"
    if raw_stage.startswith("pilot_"):
        return "retained_pilot"
    return "unclassified"


def _contains_key(value: Any, prohibited: set) -> bool:
    if isinstance(value, Mapping):
        return bool(prohibited.intersection(value)) or any(
            _contains_key(item, prohibited) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, prohibited) for item in value)
    return False


def _ledger_path(results_root: Path, episode_path: Path, episode: Mapping[str, Any]) -> Path:
    candidate = episode.get("event_ledger_path")
    if candidate is None and isinstance(episode.get("summary"), Mapping):
        candidate = episode["summary"].get("event_ledger_path")
    if candidate is not None:
        return results_root / str(candidate)
    values = list(episode_path.parent.glob("events.jsonl*"))
    if len(values) != 1:
        raise FileNotFoundError("expected exactly one ledger beside %s" % episode_path)
    return values[0]


def replay_episode(results_root: Path, episode_path: Path) -> Dict[str, Any]:
    episode = read_episode_json(episode_path)
    ledger_path = _ledger_path(results_root, episode_path, episode)
    ledger = EventLedger.read_jsonl(ledger_path)
    expected_sha = episode.get("event_ledger_sha256")
    if expected_sha is None and isinstance(episode.get("summary"), Mapping):
        expected_sha = episode["summary"].get("event_ledger_sha256")
    sha_match = expected_sha is None or sha256_file(ledger_path) == expected_sha
    expected_digest = episode.get("event_ledger_digest")
    if expected_digest is None and isinstance(episode.get("summary"), Mapping):
        expected_digest = episode["summary"].get("event_ledger_digest")
    digest_match = expected_digest is None or ledger.digest() == expected_digest
    metrics = [value.payload for value in ledger.events if value.kind == "metric"]
    summary = dict(episode.get("summary", {}))
    metric_match = bool(metrics)
    if metrics:
        for key in (
            "service_loss", "final_service_deficit", "accepted_typed_actions",
            "harmful_actions", "beneficial_actions", "sketch_messages",
            "sketch_bytes", "operator_minutes",
        ):
            if key in summary and key in metrics[-1]:
                metric_match = metric_match and bool(np.isclose(
                    float(summary[key]), float(metrics[-1][key]), atol=1e-12,
                ))
    snapshots = [value.payload for value in ledger.events if value.kind == "v6_panel_snapshot"]
    resource_initial = dict(snapshots[0].get("resource_initial", {})) if snapshots else {}
    resource_latest: Dict[str, Dict[str, float]] = {}
    for event in ledger.events:
        if event.kind == "v6_resource_transition":
            resource_latest[str(event.payload["resource"])] = {
                key: float(event.payload[key])
                for key in ("remaining", "consumed", "transferred", "losses")
            }
    residuals: Dict[str, float] = {}
    for resource, initial in resource_initial.items():
        values = resource_latest.get(resource, {
            "remaining": float(initial), "consumed": 0.0,
            "transferred": 0.0, "losses": 0.0,
        })
        residuals[resource] = float(
            float(initial) - values["remaining"] - values["consumed"]
            - values["transferred"] - values["losses"]
        )
    conservation_residual = max([abs(value) for value in residuals.values()] or [0.0])
    privacy_ok = True
    for event in ledger.events:
        if event.kind in ("v6_private_observation", "v6_belief_update", "llm_request", "llm_structured_response"):
            privacy_ok = privacy_ok and event.private_to is not None
        if event.kind == "v6_consensus_state":
            privacy_ok = privacy_ok and event.private_to == "evaluator"
        if event.kind == "v6_operational_proposal":
            deployable = event.payload.get("deployable_context", {})
            privacy_ok = privacy_ok and not _contains_key(deployable, PROHIBITED_DEPLOYABLE_KEYS)
    finite = all(
        np.isfinite(float(value))
        for value in metrics[-1].values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ) if metrics else False
    mismatch_reasons = []
    if not sha_match:
        mismatch_reasons.append("ledger_sha256")
    if not digest_match:
        mismatch_reasons.append("ledger_digest")
    if not metric_match:
        mismatch_reasons.append("metric_regeneration")
    if conservation_residual > 1e-9:
        mismatch_reasons.append("conservation")
    if not privacy_ok:
        mismatch_reasons.append("privacy")
    if not finite:
        mismatch_reasons.append("nonfinite")
    return {
        "episode_path": str(episode_path.relative_to(results_root)),
        "ledger_path": str(ledger_path.relative_to(results_root)),
        "run_id": episode.get("run_id", summary.get("run_id", episode_path.parent.name)),
        "evidence_category": _evidence_category(results_root, episode_path),
        "events": len(ledger.events),
        "ledger_sha256_match": bool(sha_match),
        "ledger_digest_match": bool(digest_match),
        "metric_regeneration_match": bool(metric_match),
        "privacy_boundary_pass": bool(privacy_ok),
        "finite_metrics": bool(finite),
        "maximum_reconstructed_conservation_residual": conservation_residual,
        "status": "pass" if not mismatch_reasons else "mismatch",
        "mismatch_reasons": ";".join(mismatch_reasons),
    }


def replay_all(results_root: Path) -> Dict[str, Any]:
    paths = list((results_root / "raw").glob("**/episode.json"))
    paths.extend((results_root / "raw").glob("**/episode.json.gz"))
    paths = sorted(paths)
    rows = [replay_episode(results_root, path) for path in paths]
    destination = results_root / "reproducibility" / "replay"
    write_csv(destination / "ledger_replay_audit.csv", rows)
    mismatches = [value for value in rows if value["status"] != "pass"]
    formal = [
        value for value in rows
        if value["evidence_category"] == "formal_or_qualified"
    ]
    formal_mismatches = [value for value in formal if value["status"] != "pass"]
    retained_pilots = [
        value for value in rows
        if value["evidence_category"] == "retained_pilot"
    ]
    unclassified = [
        value for value in rows
        if value["evidence_category"] == "unclassified"
    ]
    report = {
        "episodes_replayed": len(rows),
        "replay_mismatches": len(mismatches),
        "maximum_conservation_residual": max(
            [value["maximum_reconstructed_conservation_residual"] for value in rows] or [0.0]
        ),
        "privacy_failures": sum(not value["privacy_boundary_pass"] for value in rows),
        "nonfinite_failures": sum(not value["finite_metrics"] for value in rows),
        "formal_episodes_replayed": len(formal),
        "formal_replay_mismatches": len(formal_mismatches),
        "formal_maximum_conservation_residual": max(
            [value["maximum_reconstructed_conservation_residual"] for value in formal]
            or [0.0]
        ),
        "formal_privacy_failures": sum(
            not value["privacy_boundary_pass"] for value in formal
        ),
        "formal_nonfinite_failures": sum(
            not value["finite_metrics"] for value in formal
        ),
        "retained_pilot_episodes": len(retained_pilots),
        "retained_pilot_mismatches": sum(
            value["status"] != "pass" for value in retained_pilots
        ),
        "unclassified_episodes": len(unclassified),
        "formal_status": (
            "pass" if not formal_mismatches and not unclassified else "fail"
        ),
        "all_retained_evidence_status": (
            "pass" if not mismatches else "contains_documented_failures"
        ),
    }
    atomic_json(destination / "replay_summary.json", report)
    return report
