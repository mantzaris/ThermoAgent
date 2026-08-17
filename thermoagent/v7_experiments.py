"""Restartable V7 episode execution with coupled dynamic counterfactuals."""

from __future__ import annotations

import gzip
import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .events import EventLedger, sha256_file
from .v5_experiments import atomic_json, source_checksum, write_csv
from .v7_base import V7CoupledEnvironment
from .v7_humanitarian import HumanitarianV7Environment
from .v7_io import compressed_path, episode_artifacts, read_json_artifact
from .v7_policies import V7SelectiveController, decision_key, risk_score
from .v7_types import V7RiskContext, V7StructuredDecision
from .v7_utility import UtilityRestorationV7Environment


MODEL_IDENTIFIER = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_environment(
    application: str,
    complexity: str,
    coupling: str,
    fragmentation: str,
    network_disruption: str,
    topology_family: str,
    environment_seed: int,
    information_condition: str = "private_fragmented",
    sketch_policy: str = "event_triggered",
    operational_communication_policy: str = "agent_event_triggered",
) -> V7CoupledEnvironment:
    values = (
        application, complexity, coupling, fragmentation, network_disruption,
        topology_family, environment_seed, information_condition, sketch_policy,
        operational_communication_policy,
    )
    if application == "humanitarian":
        return HumanitarianV7Environment(*values)
    if application == "utility_restoration":
        return UtilityRestorationV7Environment(*values)
    raise ValueError("V7 primary application must be humanitarian or utility_restoration")


def _service_loss(environment: V7CoupledEnvironment) -> float:
    return float(environment.metrics()["service_loss"])


def evaluator_counterfactual(
    environment: V7CoupledEnvironment,
    decision: V7StructuredDecision,
    step: int,
    maximum_window: int = 12,
) -> Dict[str, Any]:
    """Replay action/no-action branches from identical dynamic state and tape."""
    action_branch = deepcopy(environment)
    no_action_branch = deepcopy(environment)
    forced = replace(decision, delegation_action="execute_autonomously")
    result = action_branch.validate_and_schedule(forced, step)
    before_action = _service_loss(action_branch)
    before_no_action = _service_loss(no_action_branch)
    end = min(environment.spec.horizon, step + max(2, int(maximum_window)))
    for future_step in range(step + 1, end):
        action_branch.deliver_messages(future_step)
        no_action_branch.deliver_messages(future_step)
        action_branch.advance_domain(future_step)
        no_action_branch.advance_domain(future_step)
    action_increment = _service_loss(action_branch) - before_action
    no_action_increment = _service_loss(no_action_branch) - before_no_action
    loss_reduction = float(no_action_increment - action_increment)
    action_cost = (
        0.03 * float(decision.proposal.quantity_or_capacity)
        if bool(result.get("accepted_physical_action", False)) else 0.0
    )
    causal_utility = loss_reduction - action_cost
    output = {
        "accepted": bool(result.get("accepted_physical_action", False)),
        "stochastic_tape_digest_action": action_branch.stochastic_tape_digest,
        "stochastic_tape_digest_no_action": no_action_branch.stochastic_tape_digest,
        "action_loss_increment": action_increment,
        "no_action_loss_increment": no_action_increment,
        "loss_reduction": loss_reduction,
        "action_cost": action_cost,
        "causal_utility": causal_utility,
        "beneficial": causal_utility > 1e-9,
        "harmful": causal_utility < -1e-9,
        "window_end": end,
    }
    environment.ledger.append(
        step, "v7_counterfactual_branch", "evaluator",
        {
            "agent_id": decision.proposal.agent_id,
            "target": decision.proposal.target_asset_or_location,
            "action": decision.proposal.proposed_operational_action,
            **output,
        },
        private_to="evaluator",
    )
    return output


def run_episode(
    application: str,
    complexity: str,
    coupling: str,
    fragmentation: str,
    network_disruption: str,
    topology_family: str,
    environment_seed: int,
    controller: V7SelectiveController,
    information_condition: str = "private_fragmented",
    sketch_policy: str = "event_triggered",
    results_root: Optional[Path] = None,
    stage: str = "pilot",
    counterfactual_limit_per_epoch: int = 4,
    resume: bool = True,
    operational_communication_policy: str = "agent_event_triggered",
) -> Dict[str, Any]:
    environment = make_environment(
        application, complexity, coupling, fragmentation, network_disruption,
        topology_family, environment_seed, information_condition, sketch_policy,
        operational_communication_policy,
    )
    run_id = "v7-%s-%s-%s-%s-%s-%s-%s-%s-%d-%s-%s-%s" % (
        stage, application, complexity, coupling, fragmentation,
        network_disruption, topology_family, information_condition,
        environment_seed, controller.method, sketch_policy,
        operational_communication_policy,
    )
    if results_root is not None and resume:
        existing = results_root / "raw" / stage / run_id / "episode.json"
        if existing.exists() or compressed_path(existing).exists():
            return read_json_artifact(existing)
    started = utc_now()
    candidates: List[Dict[str, Any]] = []
    operator_minutes = 0.0
    operator_escalations = 0
    eligible_operational_proposals = 0
    for step in range(environment.spec.horizon):
        environment.deliver_messages(step)
        environment.advance_domain(step)
        if step not in environment.spec.decision_steps:
            continue
        environment.deliver_private_observations(step)
        if step == environment.spec.decision_steps[0]:
            environment.process_commitments(step)
        environment.exchange_entropy_sketches(step)
        decisions: List[V7StructuredDecision] = []
        contexts: List[V7RiskContext] = []
        for agent_id in sorted(environment.agents):
            agent = environment.agents[agent_id]
            assets = sorted(agent.identity.asset_scope)
            focal_asset = assets[(step // environment.spec.decision_interval) % len(assets)]
            decision = agent.propose(focal_asset)
            context = environment.risk_context(decision, step)
            decisions.append(decision)
            contexts.append(context)
            environment.ledger.append(
                step, "v7_operational_proposal", agent_id,
                {"decision": decision.as_dict(), "deployable_context": context.deployable()},
                private_to=agent_id,
            )
        delegation = controller(contexts, step)
        # Development-only evaluator probes are stratified by proposed action,
        # not by outcome. This avoids repeatedly probing the first few roles
        # while preserving a prospective bounded computation budget.
        action_groups: Dict[str, List[int]] = {}
        for index, context in enumerate(contexts):
            if context.proposal.is_physical:
                action_groups.setdefault(
                    context.proposal.proposed_operational_action, []
                ).append(index)
        evaluation_order: List[int] = []
        rotation = step // max(environment.spec.decision_interval, 1)
        while action_groups and len(evaluation_order) < int(counterfactual_limit_per_epoch):
            exhausted = []
            for action in sorted(action_groups):
                values = action_groups[action]
                if values:
                    selection = rotation % len(values)
                    evaluation_order.append(values.pop(selection))
                    if len(evaluation_order) >= int(counterfactual_limit_per_epoch):
                        break
                if not values:
                    exhausted.append(action)
            for action in exhausted:
                action_groups.pop(action, None)
        evaluation_indices = set(evaluation_order)
        actionable_counterfactuals = 0
        for context_index, (decision, context) in enumerate(zip(decisions, contexts)):
            selected_delegation = delegation[decision_key(context)]
            selected = replace(decision, delegation_action=selected_delegation)
            counterfactual: Dict[str, Any] = {
                "causal_utility": 0.0, "beneficial": False, "harmful": False,
                "loss_reduction": 0.0, "accepted": False,
            }
            eligible_operational_proposals += int(context.proposal.is_physical)
            counterfactual_evaluated = bool(
                context.proposal.is_physical
                and context_index in evaluation_indices
            )
            if counterfactual_evaluated:
                counterfactual = evaluator_counterfactual(environment, selected, step)
                actionable_counterfactuals += 1
            if selected_delegation == "escalate_operator":
                operator_escalations += 1
                operator_minutes += 3.0 + float(context.local_kpis.get("delay", 0.0))
            result = environment.validate_and_schedule(selected, step)
            candidates.append({
                "run_id": run_id,
                "application": application,
                "complexity": complexity,
                "coupling": coupling,
                "fragmentation": fragmentation,
                "network_disruption": network_disruption,
                "topology_family": topology_family,
                "information_condition": information_condition,
                "sketch_policy": sketch_policy,
                "environment_seed": int(environment_seed),
                "controller": controller.method,
                "step": step,
                "agent_id": context.proposal.agent_id,
                "target": context.proposal.target_asset_or_location,
                "proposed_operational_action": context.proposal.proposed_operational_action,
                "information_action": selected.information_action,
                "communication_action": selected.communication_action,
                "delegation_action": selected.delegation_action,
                "accepted_physical_action": bool(result.get("accepted_physical_action", False)),
                "counterfactual_evaluated": counterfactual_evaluated,
                "counterfactual_causal_utility": float(counterfactual["causal_utility"]),
                "counterfactual_action_accepted": bool(counterfactual["accepted"]),
                "counterfactual_beneficial": bool(counterfactual["beneficial"]),
                "counterfactual_harmful": bool(counterfactual["harmful"]),
                "predictive_uncertainty": context.predictive_uncertainty,
                "action_value_margin": context.action_value_margin,
                "severity": context.local_kpis.get("severity", 0.0),
                "safety_risk": context.local_kpis.get("safety_risk", 0.0),
                "resource_scarcity": context.local_kpis.get("resource_scarcity", 0.0),
                "delay": context.local_kpis.get("delay", 0.0),
                "action_probability": context.proposal.action_probability,
                "action_value": context.proposal.action_value,
                "action_value_margin": context.proposal.value_margin,
                "communication_reliability": context.communication_reliability,
                "coupling_numeric": context.coupling_strength,
                "fragmentation_numeric": context.fragmentation,
                "size_normalized": context.size_normalized,
                "shannon_local": context.shannon_local,
                "pooled_uncertainty": context.pooled_uncertainty,
                "js_disagreement": context.js_disagreement,
                "jt_disagreement_0_5": context.jt_disagreement_0_5,
                "jt_disagreement_2": context.jt_disagreement_2,
                "jt_disagreement_1_5": context.jt_disagreement_1_5,
                "jt_disagreement_3": context.jt_disagreement_3,
                "graph_disagreement": context.graph_disagreement,
                "consensus": context.consensus,
                "consensus_residual": context.consensus_residual,
                "entropy_slope": context.entropy_slope,
                "disagreement_slope": context.disagreement_slope,
                "distributed_contributor_count": len(context.contributors),
                "distributed_missing_agent_count": len(context.missing_agents),
                "evaluator_distributed_estimation_error": environment.evaluator_estimation_errors[
                    (int(step), context.proposal.agent_id, str(context.proposal.target_asset_or_location))
                ],
                "risk_kpi_confidence": risk_score("kpi_confidence", context),
                "risk_predictive_uncertainty": risk_score("predictive_uncertainty", context),
                "risk_shannon_js": risk_score("shannon_js", context),
                "risk_generalized_tsallis_gini": risk_score("generalized_tsallis_gini", context),
                "risk_graph_disagreement": risk_score("graph_disagreement", context),
                "risk_combined_generalized_entropic": risk_score("combined_generalized_entropic", context),
            })
    metrics = environment.metrics()
    conservation = environment.conservation_report()
    privacy = environment.privacy_audit()
    environment.ledger.append(
        environment.spec.horizon, "v7_conservation_audit", "auditor",
        conservation, private_to="evaluator",
    )
    environment.ledger.append(
        environment.spec.horizon, "v7_privacy_audit", "auditor",
        privacy, private_to="evaluator",
    )
    metrics.update({
        "run_id": run_id,
        "application": application,
        "complexity": complexity,
        "coupling": coupling,
        "fragmentation": fragmentation,
        "network_disruption": network_disruption,
        "topology_family": topology_family,
        "information_condition": information_condition,
        "sketch_policy": sketch_policy,
        "operational_communication_policy": operational_communication_policy,
        "environment_seed": int(environment_seed),
        "controller": controller.method,
        "agent_count": environment.spec.agent_count,
        "operational_node_count": environment.spec.operational_nodes,
        "horizon": environment.spec.horizon,
        "decision_epochs": len(environment.spec.decision_steps),
        "operator_escalations": operator_escalations,
        "operator_minutes": operator_minutes,
        "eligible_operational_proposals": eligible_operational_proposals,
        "autonomous_action_coverage": float(
            metrics.get("physical_actions", 0) / max(eligible_operational_proposals, 1)
        ),
        "privacy_boundary_pass": bool(privacy["pass"]),
        "distributed_estimation_mae": float(np.mean(
            list(environment.evaluator_estimation_errors.values())
        )) if environment.evaluator_estimation_errors else 0.0,
        "baseline_calibration_sketch_messages": int(
            environment.sketch_messages_by_step.get(0, 0)
        ),
        "post_disruption_sketch_messages": int(sum(
            count for step, count in environment.sketch_messages_by_step.items()
            if step >= getattr(environment, "disruption_step", 0)
        )),
        "stochastic_tape_digest": environment.stochastic_tape_digest,
        "event_count": len(environment.ledger.events),
        "started_at": started,
        "completed_at": utc_now(),
    })
    environment.ledger.append(
        environment.spec.horizon, "metric", "evaluator", metrics,
        private_to="evaluator",
    )
    output = {
        "summary": metrics,
        "candidates": candidates,
        "causal_chains": environment.causal_chains,
        "edge_message_counts": [
            {"sender": key[0], "recipient": key[1], "messages": count}
            for key, count in sorted(environment.edge_message_counts.items())
        ],
        "event_ledger_digest": environment.ledger.digest(),
    }
    if results_root is not None:
        run_dir = results_root / "raw" / stage / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "events.jsonl.gz"
        ledger_sha = environment.ledger.write_jsonl(ledger_path)
        output["event_ledger_path"] = str(ledger_path.relative_to(results_root))
        output["event_ledger_sha256"] = ledger_sha
        atomic_json(run_dir / "episode.json", output)
        write_csv(run_dir / "candidate_decisions.csv", candidates)
        write_csv(run_dir / "edge_message_counts.csv", output["edge_message_counts"])
    return output


def aggregate_stage(results_root: Path, stage: str) -> Dict[str, Any]:
    episode_paths = episode_artifacts(results_root / "raw" / stage)
    summaries: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for path in episode_paths:
        value = read_json_artifact(path)
        summaries.append(dict(value["summary"]))
        candidates.extend(value.get("candidates", []))
    destination = results_root / stage
    write_csv(destination / "episode_summary.csv", summaries)
    write_csv(destination / "candidate_decisions.csv", candidates)
    report = {
        "stage": stage,
        "episodes": len(summaries),
        "candidate_decisions": len(candidates),
        "applications": sorted(set(value["application"] for value in summaries)),
        "maximum_conservation_residual": max(
            [float(value["maximum_conservation_residual"]) for value in summaries] or [0.0]
        ),
        "privacy_failures": sum(not bool(value["privacy_boundary_pass"]) for value in summaries),
    }
    atomic_json(destination / "execution_summary.json", report)
    return report
