"""Restartable V5 development execution and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .events import sha256_file
from .v5_environment import V5PanelEnvironment, payload_digest


MODEL_IDENTIFIER = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PROMPT_REVISION = "thermohitl-v5-independent-json-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def source_checksum(repository: Path) -> str:
    digest = hashlib.sha256()
    roots = (repository / "thermoagent", repository / "configs", repository / "scripts")
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".sh", ".toml"}:
                digest.update(str(path.relative_to(repository)).encode("utf-8"))
                digest.update(path.read_bytes())
    return digest.hexdigest()


def git_metadata(repository: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=str(repository), check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit": os.environ.get("THERMO_SOURCE_COMMIT", "filtered_source_bundle"),
            "branch": os.environ.get("THERMO_SOURCE_BRANCH", "filtered_source_bundle"),
            "dirty": False,
            "git_metadata_filtered": True,
        }


def protocol_checksum(repository: Path) -> str:
    return sha256_file(repository / "configs" / "human_operator_v5_development.yaml")


def _resource_key(action: str) -> Optional[str]:
    return {
        "verify": "verification_slots",
        "request_peer_evidence": "verification_slots",
        "authorize_emergency_resource": "emergency_resources",
        "deploy_repair_capacity": "repair_capacity",
        "reroute_or_reconfigure": "routing_authorizations",
        "isolate_or_quarantine": "isolation_authorizations",
    }.get(action)


def run_panel(
    application: str,
    regime: str,
    information_condition: str,
    seed: int,
    sketch_policy: str = "event_triggered",
) -> Tuple[V5PanelEnvironment, Dict[str, Any], List[Dict[str, Any]]]:
    environment = V5PanelEnvironment(
        application=application,
        regime=regime,
        information_condition=information_condition,
        seed=int(seed),
        sketch_policy=sketch_policy,
    )
    candidates = environment.candidate_rows()
    no_communication = environment.autonomous_outcome(False)
    fixed_communication = environment.autonomous_outcome(True)

    best_by_incident: List[Dict[str, Any]] = []
    for incident_id in environment.incidents:
        values = [row for row in candidates if row["incident_id"] == incident_id]
        best_by_incident.append(max(values, key=lambda row: (float(row["causal_effect"]), row["action"])))
    oracle_selected = sorted(
        best_by_incident,
        key=lambda row: (float(row["causal_effect"]), row["incident_id"]),
        reverse=True,
    )[: environment.operator_budget]
    before_state = {
        "resources": dict(environment.resource_initial),
        "loss": float(fixed_communication["loss"]),
    }
    operator_effect = 0.0
    chains = 0
    harmful = 0
    operator_minutes = 0.0
    for row in oracle_selected:
        if float(row["causal_effect"]) <= 0.0:
            continue
        incident_id = str(row["incident_id"])
        action = str(row["action"])
        resource_key = _resource_key(action)
        if resource_key is not None:
            environment.resource_used[resource_key] += 1.0
        operator_effect += float(row["causal_effect"])
        operator_minutes += float(row["operator_minutes"])
        harmful += int(bool(row["harmful"]))
        chain_complete = bool(row["accepted_action"] and row["reached_next_stage"] and row["reached_service"])
        chains += int(chain_complete)
        step = environment.incidents[incident_id].disruption_step
        environment.ledger.append(
            step, "attention_allocation", "simulated_operator",
            {"incident_id": incident_id, "action": action, "budget": environment.operator_budget, "policy": "bounded_oracle_development"},
        )
        environment.ledger.append(
            step + 1, "operator_action", "simulated_operator",
            {"incident_id": incident_id, "action": action, "bounded": True, "oracle_upper_bound": True},
        )
        environment.ledger.append(
            step + 1, "intervention_causal_stage", "simulator",
            {
                "incident_id": incident_id,
                "action": action,
                "accepted_action": bool(row["accepted_action"]),
                "reached_next_stage": bool(row["reached_next_stage"]),
                "reached_service": bool(row["reached_service"]),
                "primary_outcome_changed": float(row["causal_effect"]) != 0.0,
                "causal_chain_complete": chain_complete,
            },
        )
    conservation = environment.conservation_report()
    after_state = {
        "resources": {
            key: float(environment.resource_initial[key] - environment.resource_used[key])
            for key in environment.resource_initial
        },
        "used": dict(environment.resource_used),
        "loss": float(before_state["loss"] - operator_effect),
    }
    environment.ledger.append(
        max(value.disruption_step for value in environment.incidents.values()) + 4,
        "v5_state_transition", "simulator",
        {
            "before": before_state,
            "before_digest": payload_digest(before_state),
            "after": after_state,
            "after_digest": payload_digest(after_state),
            "conservation": conservation,
            "operator_effect": operator_effect,
        },
    )
    summary = {
        **environment.summary(),
        "run_id": "v5-%s-%s-%s-%s-e%d" % (
            application, regime, information_condition, sketch_policy, int(seed),
        ),
        "status": "complete" if conservation["feasible"] else "failed",
        "no_communication_loss": float(no_communication["loss"]),
        "fixed_communication_loss": float(fixed_communication["loss"]),
        "coordination_loss_reduction": float(no_communication["loss"] - fixed_communication["loss"]),
        "coordination_changed_outcome": bool(abs(no_communication["loss"] - fixed_communication["loss"]) > 1e-12),
        "fixed_operational_messages": int(fixed_communication["operational_messages"]),
        "fixed_operational_bytes": int(fixed_communication["operational_bytes"]),
        "fixed_accepted_actions": int(fixed_communication["accepted_actions"]),
        "fixed_service_reaching_actions": int(fixed_communication["service_reaching_actions"]),
        "fixed_negotiations": int(fixed_communication["negotiations"]),
        "fixed_commitment_revisions": int(fixed_communication["commitment_revisions"]),
        "bounded_oracle_loss": float(after_state["loss"]),
        "bounded_oracle_effect": float(operator_effect),
        "bounded_oracle_interventions": int(sum(float(row["causal_effect"]) > 0 for row in oracle_selected)),
        "bounded_oracle_operator_minutes": float(operator_minutes),
        "bounded_oracle_harmful": int(harmful),
        "complete_causal_chains": int(chains),
        "candidate_count": len(candidates),
        "beneficial_candidates": sum(int(bool(row["beneficial"])) for row in candidates),
        "harmful_candidates": sum(int(bool(row["harmful"])) for row in candidates),
        "event_count": len(environment.ledger.events),
        "event_ledger_digest": environment.ledger.digest(),
    }
    return environment, summary, candidates


def write_panel(
    repository: Path,
    results_root: Path,
    stage: str,
    environment: V5PanelEnvironment,
    summary: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    run_id = str(summary["run_id"])
    run_root = results_root / "raw" / stage / run_id
    episode_path = run_root / "episode.json"
    ledger_path = run_root / "events.jsonl.gz"
    manifest_path = results_root / "manifests" / stage / (run_id + ".json")
    if episode_path.exists() or ledger_path.exists() or manifest_path.exists():
        raise FileExistsError("V5 output already exists: %s" % run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    ledger_sha = environment.ledger.write_jsonl(ledger_path)
    episode = {
        "study": "ThermoHITL v5",
        "stage": stage,
        "summary": dict(summary),
        "candidate_interventions": list(candidates),
        "event_ledger_digest": environment.ledger.digest(),
        "stochastic_tape_digest": environment.stochastic_tape_digest,
        "simulated_operator": True,
        "real_human_participants": False,
    }
    atomic_json(episode_path, episode)
    git = git_metadata(repository)
    manifest = {
        "run_id": run_id,
        "study": "ThermoHITL v5",
        "stage": stage,
        "application": summary["application"],
        "regime": summary["regime"],
        "information_condition": summary["information_condition"],
        "environment_seed": int(summary["environment_seed"]),
        "sketch_policy": summary["sketch_policy"],
        "source_checksum": source_checksum(repository),
        "protocol_checksum": protocol_checksum(repository),
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "dirty_tree": git["dirty"],
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "prompt_revision": PROMPT_REVISION,
        "planner": "deterministic_independent_engineering_control",
        "simulated_operator": True,
        "real_human_participants": False,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "event_count": int(summary["event_count"]),
        "event_ledger_digest": environment.ledger.digest(),
        "stochastic_tape_digest": environment.stochastic_tape_digest,
        "episode_sha256": sha256_file(episode_path),
        "ledger_sha256": ledger_sha,
        "completion_status": summary["status"],
        "generated_at": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def run_matrix(
    repository: Path,
    results_root: Path,
    stage: str,
    applications: Sequence[str],
    regimes: Sequence[str],
    information_conditions: Sequence[str],
    seeds: Sequence[int],
    sketch_policies: Sequence[str] = ("event_triggered",),
    resume: bool = True,
) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    completed = 0
    resumed = 0
    for application in applications:
        for regime in regimes:
            for condition in information_conditions:
                for sketch_policy in sketch_policies:
                    for seed in seeds:
                        run_id = "v5-%s-%s-%s-%s-e%d" % (
                            application, regime, condition, sketch_policy, int(seed),
                        )
                        episode_path = results_root / "raw" / stage / run_id / "episode.json"
                        if resume and episode_path.is_file():
                            episode = json.loads(episode_path.read_text(encoding="utf-8"))
                            summaries.append(dict(episode["summary"]))
                            candidate_rows.extend(episode["candidate_interventions"])
                            resumed += 1
                            continue
                        try:
                            environment, summary, candidates = run_panel(
                                application, regime, condition, int(seed), sketch_policy,
                            )
                            write_panel(repository, results_root, stage, environment, summary, candidates)
                            summaries.append(summary)
                            candidate_rows.extend(candidates)
                            completed += 1
                        except Exception as error:
                            failure = {
                                "run_id": run_id,
                                "application": application,
                                "regime": regime,
                                "information_condition": condition,
                                "sketch_policy": sketch_policy,
                                "environment_seed": int(seed),
                                "failure_type": type(error).__name__,
                                "failure_reason": str(error),
                            }
                            failures.append(failure)
                            atomic_json(results_root / "manifests" / stage / (run_id + ".failed.json"), failure)
    stage_root = results_root / "development" / stage
    write_csv(stage_root / "episode_summary.csv", summaries)
    write_csv(stage_root / "candidate_interventions.csv", candidate_rows)
    if failures:
        write_csv(results_root / "negative_results" / (stage + "_failed_runs.csv"), failures)
    report = {
        "stage": stage,
        "completed_now": completed,
        "resumed_existing": resumed,
        "episodes": len(summaries),
        "candidate_rows": len(candidate_rows),
        "failures": len(failures),
        "finished_at": utc_now(),
    }
    atomic_json(stage_root / "run_report.json", report)
    return report
