"""Restartable V8 episodes with paired communication-policy isolation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol

import numpy as np

from .v5_experiments import atomic_json, write_csv
from .v7_experiments import make_environment
from .v7_types import DELEGATION_ACTIONS, V7StructuredDecision
from .v8_monitoring import V8BeliefNetwork
from .v8_io import write_csv_gzip, write_event_ledger_xz, write_json_gzip
from .v8_trigger import TriggerConfig
from .events import EventLedger


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DecentralizedActionPolicy(Protocol):
    policy_id: str

    def decide(
        self,
        *,
        agent: Any,
        asset: str,
        distributed_estimate: Mapping[str, Any],
        step: int,
    ) -> V7StructuredDecision:
        ...


class FrozenLocalRulePolicy:
    """Independent V7 local agent policy retained as an engineering baseline."""

    policy_id = "frozen_local_rule"

    def decide(
        self,
        *,
        agent: Any,
        asset: str,
        distributed_estimate: Mapping[str, Any],
        step: int,
    ) -> V7StructuredDecision:
        # No evaluator field is read. Delivered sketches have already updated
        # this agent's own private belief through its explicit inbox.
        return agent.propose(asset)


AUTHORIZED_ESTIMATE_FIELDS = {
    "step", "recipient", "asset", "contributors", "scoped_agents",
    "missing_agents", "maximum_age", "mean_age",
    "distributed_pooled_belief", "distributed_disagreement",
    "distributed_disrupted_probability",
}


def authorized_distributed_estimate(
    estimate: Mapping[str, Any],
) -> Dict[str, Any]:
    """Remove evaluator-only scoring fields before any policy call."""
    missing = AUTHORIZED_ESTIMATE_FIELDS - set(estimate)
    if missing:
        raise ValueError("distributed estimate lacks deployable fields: %s" % sorted(missing))
    return {key: estimate[key] for key in sorted(AUTHORIZED_ESTIMATE_FIELDS)}


def _trigger_digest(config: TriggerConfig) -> str:
    payload = asdict(config)
    payload["weights"] = dict(sorted(config.weights.items()))
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _detection_metrics(
    rows: List[Dict[str, Any]], horizon: int, threshold: float = 0.50,
) -> Dict[str, Any]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["recipient"], row["asset"]), []).append(row)
    delays: List[float] = []
    false_alarms = 0
    true_positive_groups = 0
    detected_groups = 0
    for values in groups.values():
        ordered = sorted(values, key=lambda value: int(value["step"]))
        disrupted = [int(value["step"]) for value in ordered if value["evaluator_disrupted"]]
        detections = [
            int(value["step"])
            for value in ordered
            if float(value["distributed_disrupted_probability"]) >= float(threshold)
        ]
        if not disrupted:
            false_alarms += int(bool(detections))
            continue
        true_positive_groups += 1
        onset = min(disrupted)
        false_alarms += int(any(value < onset for value in detections))
        after = [value for value in detections if value >= onset]
        if after:
            detected_groups += 1
            delays.append(float(min(after) - onset))
        else:
            delays.append(float(horizon - onset))
    return {
        "detection_groups": true_positive_groups,
        "detected_groups": detected_groups,
        "detection_recall": float(detected_groups / max(true_positive_groups, 1)),
        "mean_detection_delay_steps": float(np.mean(delays)) if delays else 0.0,
        "maximum_detection_delay_steps": float(max(delays)) if delays else 0.0,
        "false_alarm_groups": int(false_alarms),
        "false_alarm_rate": float(false_alarms / max(len(groups), 1)),
    }


def _estimation_metrics(
    rows: List[Dict[str, Any]], recovery_step: int, horizon: int,
) -> Dict[str, Any]:
    if not rows:
        return {
            "normalized_time_integrated_estimation_error": 0.0,
            "disagreement_time_integrated_error": 0.0,
            "primary_distributed_state_error": 0.0,
            "primary_distributed_state_error_p95": 0.0,
            "pointwise_estimation_mae_p90": 0.0,
            "pointwise_estimation_mae_p95": 0.0,
            "pointwise_estimation_mae_max": 0.0,
            "mean_message_age": 0.0,
            "maximum_message_age": 0.0,
            "stale_belief_rows": 0,
            "consensus_recovery_steps": 0.0,
        }
    belief_errors = np.asarray([float(row["belief_mae"]) for row in rows])
    disagreement_errors = np.asarray([
        float(row["disagreement_absolute_error"]) for row in rows
    ])
    primary_errors = 0.5 * belief_errors + 0.5 * disagreement_errors
    ages = np.asarray([float(row["maximum_age"]) for row in rows])
    by_step: Dict[int, List[float]] = {}
    for row in rows:
        if int(row["step"]) >= int(recovery_step):
            by_step.setdefault(int(row["step"]), []).append(
                float(row["disagreement_absolute_error"])
            )
    recovered = [
        step for step, values in sorted(by_step.items())
        if float(np.mean(values)) <= 0.02
    ]
    recovery = (
        max(0, min(recovered) - int(recovery_step))
        if recovered else max(0, int(horizon) - int(recovery_step))
    )
    return {
        "normalized_time_integrated_estimation_error": float(np.mean(belief_errors)),
        "disagreement_time_integrated_error": float(np.mean(disagreement_errors)),
        "primary_distributed_state_error": float(np.mean(primary_errors)),
        "primary_distributed_state_error_p95": float(np.quantile(primary_errors, 0.95)),
        "pointwise_estimation_mae_p90": float(np.quantile(belief_errors, 0.90)),
        "pointwise_estimation_mae_p95": float(np.quantile(belief_errors, 0.95)),
        "pointwise_estimation_mae_max": float(np.max(belief_errors)),
        "mean_message_age": float(np.mean(ages)),
        "maximum_message_age": float(np.max(ages)),
        "stale_belief_rows": int(np.sum(ages > 12.0)),
        "consensus_recovery_steps": float(recovery),
    }


def run_v8_episode(
    *,
    application: str,
    complexity: str,
    coupling: str,
    fragmentation: str,
    network_disruption: str,
    topology_family: str,
    environment_seed: int,
    trigger_config: TriggerConfig,
    action_policy: Optional[DecentralizedActionPolicy] = None,
    information_condition: str = "private_fragmented",
    encoding: str = "fp16",
    maximum_hops: int = 2,
    operational_communication_policy: str = "agent_event_triggered",
    results_root: Optional[Path] = None,
    stage: str = "pilots",
    resume: bool = True,
    ledger_scope: str = "full",
) -> Dict[str, Any]:
    """Run one full dynamic panel arm from a shared deterministic tape."""
    if ledger_scope not in ("full", "dynamic_delta"):
        raise ValueError("V8 ledger_scope must be full or dynamic_delta")
    policy = action_policy or FrozenLocalRulePolicy()
    trigger_hash = _trigger_digest(trigger_config)
    run_id = (
        "v8-%s-%s-%s-%s-%s-%s-%s-%s-%d-%s-%s-%s-%s" % (
            stage, application, complexity, coupling, fragmentation,
            network_disruption, topology_family, information_condition,
            int(environment_seed), trigger_config.method, trigger_hash,
            encoding, policy.policy_id,
        )
    )
    if results_root is not None and resume:
        existing = results_root / "raw" / stage / run_id / "episode.json"
        if existing.exists():
            return json.loads(existing.read_text(encoding="utf-8"))
    started = utc_now()
    environment = make_environment(
        application, complexity, coupling, fragmentation, network_disruption,
        topology_family, int(environment_seed), information_condition,
        sketch_policy="none",
        operational_communication_policy=operational_communication_policy,
    )
    belief_network = V8BeliefNetwork(
        environment, trigger_config, encoding=encoding,
        maximum_hops=maximum_hops, seed=int(environment_seed) + 880000,
    )
    decisions: List[Dict[str, Any]] = []
    escalation_requests = 0
    accepted_physical_actions = 0
    proposed_physical_actions = 0
    for step in range(environment.spec.horizon):
        belief_network.deliver(step)
        environment.deliver_messages(step)
        environment.advance_domain(step)
        if step not in environment.spec.decision_steps:
            continue
        environment.deliver_private_observations(step)
        if step == environment.spec.decision_steps[0]:
            environment.process_commitments(step)
        belief_network.exchange(step)
        belief_network.record_estimates(step)
        for agent_id in sorted(environment.agents):
            agent = environment.agents[agent_id]
            assets = sorted(agent.private_beliefs)
            asset = assets[(step // max(environment.spec.decision_interval, 1)) % len(assets)]
            estimate = belief_network.distributed_estimate(agent_id, asset, step)
            decision = policy.decide(
                agent=agent, asset=asset,
                distributed_estimate=authorized_distributed_estimate(estimate),
                step=step,
            )
            if not isinstance(decision, V7StructuredDecision):
                raise TypeError("a V8 autonomous policy must return V7StructuredDecision")
            proposed_physical_actions += int(decision.proposal.is_physical)
            escalation_requests += int(decision.delegation_action == "escalate_operator")
            # V8 has no simulated human outcome claim. An escalation request is
            # logged and withheld rather than silently converted to an oracle action.
            executed = (
                replace(decision, delegation_action="abstain")
                if decision.delegation_action == "escalate_operator" else decision
            )
            result = dict(environment.validate_and_schedule(executed, step))
            observer = getattr(policy, "observe_result", None)
            if observer is not None:
                observer(result, step=step)
            accepted_physical_actions += int(result.get("accepted_physical_action", False))
            row = {
                "step": int(step), "agent_id": agent_id, "asset": asset,
                "policy_id": policy.policy_id,
                "scheduler": trigger_config.method,
                "proposed_operational_action": decision.proposal.proposed_operational_action,
                "information_action": decision.information_action,
                "communication_action": decision.communication_action,
                "delegation_action": decision.delegation_action,
                "accepted_physical_action": bool(result.get("accepted_physical_action", False)),
                "validation_code": result.get("validation_code"),
                "belief_contributors": estimate["contributors"],
                "belief_mae_evaluator_only": estimate["belief_mae"],
            }
            decisions.append(row)
            environment.ledger.append(
                step, "v8_policy_decision", agent_id,
                {key: value for key, value in row.items() if not key.endswith("evaluator_only")},
                private_to=agent_id,
            )
    # Flush messages that were already on wire before the horizon ended. They
    # count on wire regardless; delivery is recorded only if it occurs in this
    # bounded flush window.
    for step in range(environment.spec.horizon, environment.spec.horizon + 8):
        belief_network.deliver(step)
        environment.deliver_messages(step)
    domain_metrics = dict(environment.metrics())
    conservation = dict(environment.conservation_report())
    privacy = dict(environment.privacy_audit())
    environment.ledger.append(
        environment.spec.horizon, "v8_conservation_audit", "auditor",
        conservation, private_to="evaluator",
    )
    environment.ledger.append(
        environment.spec.horizon, "v8_privacy_audit", "auditor",
        privacy, private_to="evaluator",
    )
    estimation = _estimation_metrics(
        belief_network.estimation_rows,
        int(getattr(environment, "recovery_step", environment.spec.horizon)),
        environment.spec.horizon,
    )
    detection = _detection_metrics(
        belief_network.estimation_rows, environment.spec.horizon,
    )
    accounting = belief_network.accounting()
    actionable = max(int(domain_metrics.get("actionable_opportunities", 0)), 1)
    normalized_reward = float(
        float(domain_metrics.get("net_causal_utility", 0.0)) / actionable
        - float(domain_metrics.get("service_loss", 0.0))
        / max(environment.spec.horizon * environment.spec.agent_count, 1)
    )
    delegation_counts = {
        action: sum(value["delegation_action"] == action for value in decisions)
        for action in DELEGATION_ACTIONS
    }
    summary = {
        **domain_metrics,
        **estimation,
        **detection,
        **accounting,
        "run_id": run_id,
        "stage": stage,
        "application": application,
        "complexity": complexity,
        "coupling": coupling,
        "fragmentation": fragmentation,
        "network_disruption": network_disruption,
        "topology_family": topology_family,
        "information_condition": information_condition,
        "environment_seed": int(environment_seed),
        "scheduler": trigger_config.method,
        "trigger_configuration_digest": trigger_hash,
        "encoding": encoding,
        "action_policy_id": policy.policy_id,
        "estimator_target": "pooled_current_private_evidence_pre_exchange",
        "agent_count": environment.spec.agent_count,
        "operational_node_count": environment.spec.operational_nodes,
        "horizon": environment.spec.horizon,
        "decision_epochs": len(environment.spec.decision_steps),
        "proposed_physical_actions": proposed_physical_actions,
        "accepted_physical_actions_v8": accepted_physical_actions,
        "operator_escalation_requests": escalation_requests,
        "policy_delegation_diversity": sum(
            count > 0 for count in delegation_counts.values()
        ),
        **{
            "policy_delegation_%s" % action: int(count)
            for action, count in delegation_counts.items()
        },
        "normalized_autonomous_reward": normalized_reward,
        "maximum_conservation_residual": float(conservation["maximum_residual"]),
        "conservation_feasible": bool(conservation["feasible"]),
        "privacy_boundary_pass": bool(privacy["pass"]),
        "stochastic_tape_digest": environment.stochastic_tape_digest,
        "event_count": len(environment.ledger.events),
        "event_ledger_scope": ledger_scope,
        "started_at": started,
        "completed_at": utc_now(),
    }
    finisher = getattr(policy, "finish_episode", None)
    if finisher is not None:
        finisher(
            completed_actions=list(getattr(environment, "completed_actions", [])),
            summary=summary,
        )
    environment.ledger.append(
        environment.spec.horizon, "metric", "evaluator", summary,
        private_to="evaluator",
    )
    stored_ledger = environment.ledger
    if ledger_scope == "dynamic_delta":
        stored_ledger = EventLedger()
        for event in environment.ledger.events:
            keep = (
                event.kind == "disruption"
                or event.kind == "metric"
                or event.kind.startswith("v8_")
                or event.kind.startswith("v7_action_")
                or event.kind.startswith("v7_resource_")
                or event.kind.startswith("v7_service_")
                or event.kind.startswith("v7_cascade_")
                or event.kind.startswith("v7_message_")
                or event.kind in (
                    "v7_communication_action", "v7_delegation_decision",
                    "v7_information_action", "v7_commitment_transition",
                )
            )
            if keep:
                stored_ledger.append(
                    event.step, event.kind, event.actor, event.payload,
                    private_to=event.private_to,
                )
    output: Dict[str, Any] = {
        "summary": summary,
        "decisions": decisions,
        "estimation_rows": belief_network.estimation_rows,
        "delivery_rows": belief_network.delivery_rows,
        "trigger_rows": belief_network.trigger_rows,
        "edge_communication": [
            {
                "sender": sender, "recipient": recipient,
                "attempted_messages": belief_network.edge_attempts[(sender, recipient)],
                "transmitted_messages": count,
                "on_wire_bytes": belief_network.edge_bytes[(sender, recipient)],
            }
            for (sender, recipient), count in sorted(belief_network.edge_transmissions.items())
        ],
        "causal_chains": environment.causal_chains,
        "event_ledger_digest": stored_ledger.digest(),
        "stored_event_count": len(stored_ledger.events),
    }
    if results_root is not None:
        run_dir = results_root / "raw" / stage / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "events.jsonl.xz"
        ledger_sha = write_event_ledger_xz(ledger_path, stored_ledger)
        output["event_ledger_path"] = str(ledger_path.relative_to(results_root))
        output["event_ledger_sha256"] = ledger_sha
        tables = {
            "decisions": write_csv_gzip(run_dir / "decisions.csv.gz", decisions),
            "estimation": write_csv_gzip(
                run_dir / "estimation.csv.gz", belief_network.estimation_rows,
            ),
            "deliveries": write_csv_gzip(
                run_dir / "deliveries.csv.gz", belief_network.delivery_rows,
            ),
            "triggers": write_csv_gzip(
                run_dir / "triggers.csv.gz", belief_network.trigger_rows,
            ),
            "edge_communication": write_csv_gzip(
                run_dir / "edge_communication.csv.gz", output["edge_communication"],
            ),
            "causal_chains": write_json_gzip(
                run_dir / "causal_chains.json.gz", output["causal_chains"],
            ),
        }
        for value in tables.values():
            value["path"] = str(Path(value["path"]).relative_to(results_root))
        compact = {
            "summary": summary,
            "event_ledger_digest": output["event_ledger_digest"],
            "event_ledger_path": output["event_ledger_path"],
            "event_ledger_sha256": ledger_sha,
            "tables": tables,
        }
        atomic_json(run_dir / "episode.json", compact)
    return output


def aggregate_v8_stage(results_root: Path, stage: str) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    for path in sorted((results_root / "raw" / stage).glob("*/episode.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(dict(value["summary"]))
    write_csv(results_root / stage / "episode_summary.csv", summaries)
    report = {
        "stage": stage,
        "completed_episodes": len(summaries),
        "applications": sorted({value["application"] for value in summaries}),
        "schedulers": sorted({value["scheduler"] for value in summaries}),
        "independent_panels": len({
            (value["application"], int(value["environment_seed"])) for value in summaries
        }),
    }
    atomic_json(results_root / stage / "execution_summary.json", report)
    return report
