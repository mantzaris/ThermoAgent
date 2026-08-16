"""Real-Qwen V7 pilot with the corrected four-part decision schema."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .planners import TransformersPlanner, extract_json_object
from .v5_experiments import atomic_json, source_checksum, utc_now, write_csv
from .v7_experiments import (
    MODEL_IDENTIFIER, MODEL_REVISION, evaluator_counterfactual, make_environment,
)
from .v7_types import (
    COMMUNICATION_ACTIONS, DELEGATION_ACTIONS, INFORMATION_ACTIONS,
    V7OperationalProposal, V7StructuredDecision,
)


PROMPT_REVISION = "v7-independent-four-field-decision-v1"
PILOT_SPECIFICATIONS = (
    ("humanitarian", "small", "high", "high", "medium", "modular", 778101),
    ("humanitarian", "medium", "high", "high", "high", "small_world", 778102),
    ("utility_restoration", "small", "high", "high", "medium", "grid", 778201),
    ("utility_restoration", "medium", "high", "high", "high", "modular", 778202),
)


def formal_specifications() -> Tuple[Tuple[Any, ...], ...]:
    rows: List[Tuple[Any, ...]] = []
    for application_index, application in enumerate(("humanitarian", "utility_restoration")):
        topologies = (
            ("random_geometric", "small_world", "modular")
            if application == "humanitarian" else ("grid", "scale_free", "modular")
        )
        for index in range(20):
            information = "private_fragmented" if index < 14 else "public_shared"
            coupling = ("medium", "high", "high")[index % 3]
            fragmentation = (
                ("medium", "high", "high")[index % 3]
                if information == "private_fragmented" else "low"
            )
            disruption = ("medium", "high", "high")[(index // 2) % 3]
            rows.append((
                application, "small" if index < 4 else "medium",
                coupling, fragmentation, disruption,
                topologies[index % len(topologies)],
                788100 + 1000 * application_index + index,
                information,
            ))
    return tuple(rows)


def authorized_payload(environment: Any, agent_id: str, asset: str) -> Dict[str, Any]:
    agent = environment.agents[agent_id]
    observation = agent.vault.observation(agent_id, asset)
    return {
        "identity": asdict(agent.identity),
        "private_observation": asdict(observation),
        "private_belief": list(agent.private_beliefs[asset]),
        "private_utility": asdict(agent.utility),
        "private_memory": [asdict(value) for value in agent.vault.memory(agent_id)][-5:],
        "explicitly_delivered_messages": [asdict(value) for value in agent.inbox][-5:],
        "private_commitments": [asdict(value) for value in agent.commitments.values()],
        "allowed_operational_actions": list(observation.feasible_physical_actions),
        "information_actions": list(INFORMATION_ACTIONS),
        "communication_actions": list(COMMUNICATION_ACTIONS),
        "delegation_actions": list(DELEGATION_ACTIONS),
    }


def qwen_prompt(
    planner: TransformersPlanner, environment: Any, agent_id: str, asset: str,
) -> str:
    system = (
        "You are one persistent independent autonomous organization in an abstract "
        "humanitarian-logistics or defensive utility-restoration simulation. You see "
        "only your private observation, belief, utility, memory, commitments, and "
        "explicitly delivered messages. Never infer evaluator truth, future events, "
        "counterfactual outcomes, or another agent's private state. Choose four "
        "separate fields: (1) a role-authorized physical operational proposal, "
        "(2) an optional information action, (3) an optional communication action, "
        "and (4) execute/defer/abstain/escalate delegation. Verification and peer "
        "evidence are not physical service actions. Endless verification delays "
        "service; unsafe physical action can cause bounded harm. Correct no-action is "
        "allowed in nominal conditions, but choose a feasible consequential physical "
        "action when your private evidence and authority justify it. Escalate when "
        "evidence conflicts and stakes are high. Return exactly one JSON object with "
        "keys proposed_operational_action, target_asset_or_location, quantity_or_capacity, "
        "information_action, communication_action, delegation_action, confidence, "
        "reason_code, compact_plan_summary. Use only listed values and numeric bounds "
        "0<=quantity<=2 and 0<=confidence<=1. Do not use markdown or reveal hidden reasoning."
    )
    user = json.dumps(
        authorized_payload(environment, agent_id, asset),
        sort_keys=True, separators=(",", ":"),
    )
    return planner.tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def validate_qwen_decision(
    environment: Any, agent_id: str, raw: str,
) -> Tuple[Optional[V7StructuredDecision], str, Dict[str, Any]]:
    value = extract_json_object(raw)
    if not isinstance(value, dict):
        return None, "response_not_object", {}
    required = {
        "proposed_operational_action", "target_asset_or_location",
        "quantity_or_capacity", "information_action", "communication_action",
        "delegation_action", "confidence", "reason_code", "compact_plan_summary",
    }
    if not required.issubset(value):
        return None, "missing_required_field", value
    agent = environment.agents[agent_id]
    target = str(value["target_asset_or_location"])
    if target not in agent.identity.asset_scope:
        return None, "target_outside_private_scope", value
    observation = agent.vault.observation(agent_id, target)
    action = str(value["proposed_operational_action"])
    if action not in observation.feasible_physical_actions:
        return None, "operational_action_masked", value
    information = str(value["information_action"])
    communication = str(value["communication_action"])
    delegation = str(value["delegation_action"])
    if information not in INFORMATION_ACTIONS:
        return None, "information_action_invalid", value
    if communication not in COMMUNICATION_ACTIONS:
        return None, "communication_action_invalid", value
    if delegation not in DELEGATION_ACTIONS:
        return None, "delegation_action_invalid", value
    try:
        quantity = float(value["quantity_or_capacity"])
        confidence = float(value["confidence"])
    except (TypeError, ValueError):
        return None, "numeric_field_invalid", value
    if not np.isfinite(quantity) or not 0.0 <= quantity <= 2.0:
        return None, "quantity_bounds", value
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None, "confidence_bounds", value
    reason = str(value["reason_code"])
    summary = str(value["compact_plan_summary"])
    if not reason or len(reason) > 96 or not summary or len(summary) > 240:
        return None, "text_field_bounds", value
    belief = np.asarray(agent.private_beliefs[target], dtype=float)
    ordered = np.sort(belief)[::-1]
    proposal = V7OperationalProposal(
        agent_id=agent_id,
        application=environment.application,
        role=agent.identity.role,
        proposed_operational_action=action,
        target_asset_or_location=target,
        source_asset_or_location=None,
        commodity_or_resource=None,
        quantity_or_capacity=quantity,
        expected_delay=max(1, int(round(observation.local_kpis.get("delay", 1.0)))),
        action_probability=confidence,
        action_value=confidence - 0.25 * observation.local_kpis.get("resource_scarcity", 0.0),
        value_margin=float(ordered[0] - ordered[1]),
        reason_code=reason,
    )
    return V7StructuredDecision(
        proposal=proposal,
        information_action=information,
        communication_action=communication,
        delegation_action=delegation,
        confidence=confidence,
        compact_plan_summary=summary,
    ), "", value


def repair_prompt(planner: TransformersPlanner, prompt: str, raw: str, error: str) -> str:
    return (
        prompt + raw[:1000]
        + "\nThe validator rejected that object (%s). This is the one allowed repair. "
        "Return only a corrected object using the exact listed fields, actions, scope, and bounds."
        % error[:120]
    )


def _pilot_cases(environment: Any, step: int, per_epoch: int = 4) -> List[Tuple[str, str]]:
    agents = sorted(environment.agents)
    offset = (step // environment.spec.decision_interval) % len(agents)
    output = []
    for index in range(min(per_epoch, len(agents))):
        agent_id = agents[(offset + index * 3) % len(agents)]
        assets = sorted(environment.agents[agent_id].identity.asset_scope)
        asset = assets[(step // environment.spec.decision_interval) % len(assets)]
        output.append((agent_id, asset))
    return output


def run_qwen_pilot(
    repository: Path, results_root: Path,
    specifications: Optional[Sequence[Tuple[Any, ...]]] = None,
    stage: str = "qwen_schema_pilot",
) -> Dict[str, Any]:
    planner = TransformersPlanner(
        MODEL_IDENTIFIER, MODEL_REVISION,
        max_new_tokens=180, max_input_tokens=3072,
        load_in_4bit=True, seed=778888,
    )
    selected_specifications = tuple(specifications or PILOT_SPECIFICATIONS)
    environments = [make_environment(*spec) for spec in selected_specifications]
    rows: List[Dict[str, Any]] = []
    total_calls = total_prompt = total_generated = 0
    generation_seconds = 0.0
    started = time.perf_counter()
    for step in range(max(environment.spec.horizon for environment in environments)):
        active = [environment for environment in environments if step < environment.spec.horizon]
        for environment in active:
            environment.deliver_messages(step)
            environment.advance_domain(step)
            if step not in environment.spec.decision_steps:
                continue
            environment.deliver_private_observations(step)
            if step == environment.spec.decision_steps[0]:
                environment.process_commitments(step)
            environment.exchange_entropy_sketches(step)
        cases: List[Tuple[Any, str, str, str]] = []
        for environment in active:
            if step not in environment.spec.decision_steps:
                continue
            for agent_id, asset in _pilot_cases(environment, step):
                prompt = qwen_prompt(planner, environment, agent_id, asset)
                cases.append((environment, agent_id, asset, prompt))
        if not cases:
            continue
        raw: List[str] = []
        prompt_counts: List[int] = []
        generated_counts: List[int] = []
        # Bounded batches avoid padding 40 independent environments into one
        # 4090-sized generation request while preserving separate contexts.
        for batch_start in range(0, len(cases), 8):
            batch = cases[batch_start:batch_start + 8]
            batch_raw, batch_prompt, batch_generated, latency = planner._generate_prompts(
                [value[3] for value in batch]
            )
            raw.extend(batch_raw)
            prompt_counts.extend(map(int, batch_prompt))
            generated_counts.extend(map(int, batch_generated))
            generation_seconds += float(latency)
        for index, (environment, agent_id, asset, prompt) in enumerate(cases):
            response = raw[index]
            decision, error, _ = validate_qwen_decision(environment, agent_id, response)
            first_pass = decision is not None
            calls = 1
            prompt_tokens = int(prompt_counts[index])
            generated_tokens = int(generated_counts[index])
            repair_attempted = False
            if decision is None:
                repair_attempted = True
                repaired_raw, repaired_prompt, repaired_generated, repaired_latency = planner._generate_prompts([
                    repair_prompt(planner, prompt, response, error)
                ])
                generation_seconds += float(repaired_latency)
                calls += 1
                prompt_tokens += int(repaired_prompt[0])
                generated_tokens += int(repaired_generated[0])
                response = repaired_raw[0]
                decision, error, _ = validate_qwen_decision(environment, agent_id, response)
            valid_after_repair = decision is not None
            if decision is None:
                deterministic = environment.agents[agent_id].propose(asset)
                decision = V7StructuredDecision(
                    proposal=V7OperationalProposal(**{
                        **asdict(deterministic.proposal),
                        "proposed_operational_action": "no_operational_action",
                        "quantity_or_capacity": 0.0,
                        "reason_code": "one_repair_exhausted",
                    }),
                    information_action="no_information_action",
                    communication_action="no_communication_action",
                    delegation_action="abstain",
                    confidence=0.0,
                    compact_plan_summary="validator fallback",
                )
            counterfactual = (
                evaluator_counterfactual(environment, decision, step)
                if decision.proposal.is_physical else {
                    "causal_utility": 0.0, "beneficial": False,
                    "harmful": False, "accepted": False,
                }
            )
            result = environment.validate_and_schedule(decision, step)
            belief_argmax = int(np.argmax(environment.agents[agent_id].private_beliefs[asset]))
            rows.append({
                "application": environment.application,
                "complexity": environment.complexity,
                "environment_seed": environment.environment_seed,
                "step": step, "agent_id": agent_id,
                "role": environment.agents[agent_id].identity.role,
                "target": asset, "private_belief_argmax": belief_argmax,
                "proposed_operational_action": decision.proposal.proposed_operational_action,
                "information_action": decision.information_action,
                "communication_action": decision.communication_action,
                "delegation_action": decision.delegation_action,
                "confidence": decision.confidence,
                "information_condition": environment.information_condition,
                "before_disruption": bool(step < environment.disruption_step),
                "first_pass_valid": first_pass,
                "repair_attempted": repair_attempted,
                "valid_after_one_repair": valid_after_repair,
                "accepted_physical_action": bool(result.get("accepted_physical_action", False)),
                "counterfactual_action_accepted": bool(counterfactual["accepted"]),
                "causal_utility": float(counterfactual["causal_utility"]),
                "beneficial": bool(counterfactual["beneficial"]),
                "harmful": bool(counterfactual["harmful"]),
                "llm_calls": calls, "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "raw_response_not_logged": True,
            })
            total_calls += calls
            total_prompt += prompt_tokens
            total_generated += generated_tokens
    application_rows: Dict[str, Any] = {}
    for application in ("humanitarian", "utility_restoration"):
        subset = [value for value in rows if value["application"] == application]
        physical = [value for value in subset if value["counterfactual_action_accepted"]]
        environment_subset = [value for value in environments if value.application == application]
        divergent_groups = 0
        eligible_divergent_groups = 0
        grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for value in subset:
            grouped.setdefault((int(value["environment_seed"]), int(value["step"])), []).append(value)
        for values in grouped.values():
            if len({value["private_belief_argmax"] for value in values}) > 1:
                eligible_divergent_groups += 1
                divergent_groups += int(
                    len({value["proposed_operational_action"] for value in values}) > 1
                )
        per_agent_actions: Dict[str, set] = {}
        for value in subset:
            key = "%s|%s" % (value["environment_seed"], value["agent_id"])
            per_agent_actions.setdefault(key, set()).add(value["proposed_operational_action"])
        application_rows[application] = {
            "episodes": len({value["environment_seed"] for value in subset}),
            "decision_epochs": len(subset),
            "first_pass_validity": float(np.mean([value["first_pass_valid"] for value in subset])),
            "validity_after_one_repair": float(np.mean([value["valid_after_one_repair"] for value in subset])),
            "action_diversity": len({value["proposed_operational_action"] for value in subset}),
            "delegation_diversity": len({value["delegation_action"] for value in subset}),
            "physical_action_rate": float(len(physical) / max(len(subset), 1)),
            "harmful_rate_among_physical": float(np.mean([value["harmful"] for value in physical])) if physical else 0.0,
            "mean_causal_utility_among_physical": float(np.mean([value["causal_utility"] for value in physical])) if physical else 0.0,
            "beneficial_rate_among_physical": float(np.mean([value["beneficial"] for value in physical])) if physical else 0.0,
            "correct_no_action_rate_before_disruption": float(np.mean([
                value["proposed_operational_action"] == "no_operational_action"
                for value in subset if value["before_disruption"]
            ])) if any(value["before_disruption"] for value in subset) else 0.0,
            "service_reaching_actions": int(sum(
                value.metrics().get("service_reaching_actions", 0)
                for value in environment_subset
            )),
            "private_evidence_divergent_groups": eligible_divergent_groups,
            "private_evidence_behavioral_divergence_rate": float(
                divergent_groups / max(eligible_divergent_groups, 1)
            ),
            "agents_with_repeated_decision_adaptation": int(sum(
                len(actions) > 1 for actions in per_agent_actions.values()
            )),
            "evidence_requests": sum(value["information_action"] != "no_information_action" for value in subset),
            "communications": sum(value["communication_action"] != "no_communication_action" for value in subset),
            "escalations": sum(value["delegation_action"] == "escalate_operator" for value in subset),
            "abstentions": sum(value["delegation_action"] == "abstain" for value in subset),
        }
    report = {
        "stage": stage,
        "evidence_status": "development_qualification_not_primary_controller_evidence",
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "prompt_revision": PROMPT_REVISION,
        "precision": "bitsandbytes NF4 with BF16 computation",
        "episodes": len(selected_specifications),
        "decision_epochs": len(rows),
        "applications": application_rows,
        "llm_calls": total_calls,
        "prompt_tokens": total_prompt,
        "generated_tokens": total_generated,
        "generation_seconds": generation_seconds,
        "wall_seconds_including_model_load": float(time.perf_counter() - started),
        "source_checksum": source_checksum(repository),
        "generated_at": utc_now(),
        "real_qwen_agents": True,
        "real_human_participants": False,
    }
    write_csv(results_root / "qwen" / (stage + "_decisions.csv"), rows)
    atomic_json(results_root / "qwen" / (stage + "_summary.json"), report)
    return report


def run_qwen_qualification(repository: Path, results_root: Path) -> Dict[str, Any]:
    return run_qwen_pilot(
        repository, results_root, formal_specifications(),
        stage="qwen_formal_qualification",
    )
