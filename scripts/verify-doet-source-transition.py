#!/usr/bin/env python3
"""Verify behavior equivalence across the validation-to-holdout source transition.

The v2 validation started before later audit-only event fields and deterministic
serialization fixes were committed.  This command runs the same deterministic
mock-planner cases in isolated Python processes rooted at both source trees. It
compares every research-facing result section and the causal event sequence,
excluding only the two explicitly added private audit event kinds and event IDs
whose ordinal necessarily changes when those audit rows are inserted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Mapping


VALIDATION_COMMIT = "2a459e3be425c5c0e51df76fc2318bbe446c2948"
EXCLUDED_AUDIT_EVENTS = ("coordination_activity", "trigger_local_state")
METHODS = (
    "fixed_always_on",
    "doet_rule",
    "kpi_cusum_trigger",
    "learned_no_entropy",
    "doet_rl",
)


WORKER = r'''
import json
from pathlib import Path
import sys

from thermoagent.environment import ScenarioConfig
from thermoagent.experiments import source_checksum
from thermoagent.planners import MockPlanner
from thermoagent.policy import CoordinationPolicy, PPOConfig
from thermoagent.runner import EpisodeRunner, calibration_from_json

request = json.loads(sys.stdin.read())
normalizer_record = json.loads(
    Path(request["normalizers_path"]).read_text(encoding="utf-8")
)
calibration = calibration_from_json(Path(request["calibration_path"]))
trigger_parameters = {
    "trigger_type": "cusum",
    "direction": "low",
    "rho": 0.8,
    "kappa": 0.15,
    "tau_on": 1.5,
    "tau_off": 0.5,
    "tau_crisis": 3.0,
    "minimum_dwell": 2,
    "cooldown": 2,
    "propagation": "local",
    "quiet_gossip_rounds": 1,
    "targeted_gossip_rounds": 1,
    "crisis_gossip_rounds": 1,
    "quiet_gossip_period": 8,
    "targeted_gossip_period": 4,
    "crisis_gossip_period": 2,
    "quiet_decision_interval": 8,
    "targeted_decision_interval": 4,
    "crisis_decision_interval": 2,
    "max_alert_neighbors": 2,
}
configuration = ScenarioConfig(
    application="commercial",
    seed=424242,
    horizon=8,
    n_agents=10,
    private_information=1.0,
    objective_misalignment=1.0,
    communication="intermittent",
    disruption="correlated",
    decision_interval=4,
    communication_budget=300,
    topology="tri_region_bridge_v2",
)
output = {"source_checksum": source_checksum(Path.cwd()), "cases": {}}
for method in request["methods"]:
    policy = None
    if method in ("learned_no_entropy", "doet_rl"):
        policy = CoordinationPolicy(PPOConfig(), seed=1701)
    normalizers = None
    if method in ("doet_rule", "doet_rl"):
        normalizers = normalizer_record["normalizers"]
    elif method == "kpi_cusum_trigger":
        normalizers = normalizer_record["kpi_normalizers"]
    runner = EpisodeRunner(
        configuration,
        method,
        planner=MockPlanner(),
        policy=policy,
        calibration=calibration,
        deterministic_policy=True,
        monitor_window=1,
        monitor_formulation="pooled",
        trigger_config=(
            trigger_parameters if method in (
                "doet_rule", "doet_rl", "kpi_cusum_trigger"
            ) else None
        ),
        trigger_normalizers=normalizers,
        fixed_broadcast_fanout=3,
    )
    result = runner.run("source-transition-" + method)
    events = []
    for event in runner.env.ledger.events:
        if event.kind in request["excluded_event_kinds"]:
            continue
        row = event.as_dict()
        row.pop("event_id", None)
        events.append(row)
    output["cases"][method] = {
        "metrics": result.metrics,
        "agent_metrics": result.agent_metrics,
        "planner_metrics": result.planner_metrics,
        "time_series": result.time_series,
        "trajectory": result.trajectory,
        "causal_events_excluding_new_private_audit_rows": events,
    }
print(json.dumps(output, sort_keys=True, separators=(",", ":")))
'''


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def _run_worker(source: Path, request: Mapping[str, Any]) -> Dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source)
    process = subprocess.run(
        [sys.executable, "-c", WORKER],
        cwd=str(source),
        env=environment,
        input=json.dumps(request, sort_keys=True),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "source-transition worker failed for %s (exit %d): %s"
            % (source, process.returncode, process.stderr[-4000:])
        )
    return json.loads(process.stdout)


def _git_commit(source: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--current-source", type=Path, default=Path.cwd())
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--normalizers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-commit", default=VALIDATION_COMMIT)
    parser.add_argument("--expected-validation-checksum")
    parser.add_argument("--expected-current-checksum")
    arguments = parser.parse_args()

    validation_source = arguments.validation_source.resolve()
    current_source = arguments.current_source.resolve()
    calibration = arguments.calibration.resolve()
    normalizers = arguments.normalizers.resolve()
    for path in (validation_source, current_source, calibration, normalizers):
        if not path.exists():
            raise FileNotFoundError(path)
    request = {
        "calibration_path": str(calibration),
        "normalizers_path": str(normalizers),
        "methods": list(METHODS),
        "excluded_event_kinds": list(EXCLUDED_AUDIT_EVENTS),
    }
    validation = _run_worker(validation_source, request)
    current = _run_worker(current_source, request)
    checks = []
    all_equal = True
    for method in METHODS:
        section_rows = []
        for section in sorted(validation["cases"][method]):
            before_hash = _canonical_hash(
                validation["cases"][method][section]
            )
            after_hash = _canonical_hash(current["cases"][method][section])
            equal = before_hash == after_hash
            all_equal = all_equal and equal
            section_rows.append({
                "section": section,
                "validation_sha256": before_hash,
                "current_sha256": after_hash,
                "equal": equal,
            })
        checks.append({
            "method": method,
            "all_sections_equal": all(row["equal"] for row in section_rows),
            "sections": section_rows,
        })
    expected_ok = True
    if arguments.expected_validation_checksum:
        expected_ok = expected_ok and (
            validation["source_checksum"]
            == arguments.expected_validation_checksum
        )
    if arguments.expected_current_checksum:
        expected_ok = expected_ok and (
            current["source_checksum"] == arguments.expected_current_checksum
        )
    report = {
        "status": "passed" if all_equal and expected_ok else "failed",
        "purpose": (
            "Behavior-equivalence gate for the validation-source to locked-"
            "holdout-source transition; this is not research evidence."
        ),
        "validation_source": {
            "commit": arguments.validation_commit,
            "source_checksum": validation["source_checksum"],
        },
        "current_source": {
            "commit": _git_commit(current_source),
            "source_checksum": current["source_checksum"],
        },
        "calibration_sha256": _sha256_file(calibration),
        "normalizers_sha256": _sha256_file(normalizers),
        "scenario": {
            "application": "commercial",
            "environment_seed": 424242,
            "horizon": 8,
            "agents": 10,
            "communication": "intermittent",
            "disruption": "correlated",
            "topology": "tri_region_bridge_v2",
            "planner": "deterministic mock-v2",
            "policy_initialization_seed": 1701,
        },
        "methods": list(METHODS),
        "excluded_event_kinds": list(EXCLUDED_AUDIT_EVENTS),
        "excluded_event_fields": ["event_id"],
        "exclusion_justification": (
            "These private audit rows were deliberately added after validation; "
            "their insertion changes ordinal event IDs but not simulator state, "
            "agent inputs, actions, messages, tools, outcomes, or causal payloads."
        ),
        "checks": checks,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if report["status"] != "passed":
        print("DOET source-transition equivalence failed", file=sys.stderr)
        return 1
    print("DOET source-transition equivalence passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
