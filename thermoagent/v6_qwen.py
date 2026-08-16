"""Substantial sequential real-Qwen qualification for independent V6 agents."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .events import sha256_file
from .planners import TransformersPlanner, extract_json_object
from .types import MemoryRecord
from .v5_experiments import atomic_json, source_checksum, utc_now, write_csv
from .v6_environment import V6PanelEnvironment
from .v6_experiments import (
    MODEL_IDENTIFIER, MODEL_REVISION, PROMPT_REVISION, _atomic_episode,
    git_metadata, read_episode_json,
)
from .v6_types import (
    DELEGATION_ACTIONS, INCIDENT_MODES, PRIMARY_ACTION_FOR_MODE,
    SECONDARY_ACTION_FOR_MODE, V6ActionProposal, V6ToolCall,
)


QWEN_EPISODE_COUNTS = {
    "humanitarian": 60,
    "utility_restoration": 60,
    "commercial": 30,
}
QWEN_BASE_SEEDS = {
    "humanitarian": 66501,
    "utility_restoration": 66601,
    "commercial": 66701,
}
QWEN_REGIMES = (
    "isolated_physical", "telemetry_integrity", "partition",
    "compound", "ood",
)


def _prompt(
    planner: TransformersPlanner,
    environment: V6PanelEnvironment,
    agent_id: str,
    incident_id: str,
) -> str:
    agent = environment.agents[agent_id]
    payload = {
        "identity": asdict(agent.identity),
        "private_observation": asdict(agent.vault.observation(agent_id, incident_id)),
        "private_belief": list(agent.private_beliefs[incident_id]),
        "private_utility": asdict(agent.utility),
        "private_memory": [asdict(value) for value in agent.vault.memory(agent_id)][-4:],
        "explicitly_delivered_messages": [asdict(value) for value in agent.inbox][-4:],
        "private_commitments": [asdict(value) for value in agent.commitments.values()],
        "allowed_actions": list(environment.registry.allowed_actions(agent.identity.role)),
        "delegation_actions": list(DELEGATION_ACTIONS),
        "incident_scope": list(agent.identity.incident_scope),
    }
    system = (
        "You are one independent autonomous organization in an abstract logistics or defensive utility-restoration simulation. "
        "You cannot see another agent's private observation, evaluator truth, future event, or counterfactual outcome. "
        "Choose one role-authorized operational action and independently choose whether to execute, communicate, request evidence, "
        "defer, abstain, or escalate to a bounded simulated operator. Return exactly one compact JSON object with keys action, "
        "incident_id, quantity, reason_code, plan_summary, confidence, delegation. Action must be in allowed_actions; delegation "
        "must be in delegation_actions; incident_id must be in incident_scope; quantity must be 0 through 2; confidence must be "
        "0 through 1. Use private evidence and utility. Do not use markdown, invent IDs, or reveal hidden reasoning."
    )
    user = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return planner.tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def _validate(
    environment: V6PanelEnvironment,
    agent_id: str,
    raw: str,
) -> Tuple[Optional[V6ToolCall], Optional[str], Optional[float], str, Dict[str, Any]]:
    value = extract_json_object(raw)
    if not isinstance(value, dict):
        return None, None, None, "response_not_object", {}
    try:
        call = V6ToolCall(
            action=str(value["action"]),
            incident_id=str(value["incident_id"]),
            quantity=float(value.get("quantity", 1.0)),
            reason_code=str(value.get("reason_code", "qwen_private_plan")),
        )
        delegation = str(value["delegation"])
        confidence = float(value["confidence"])
    except (KeyError, TypeError, ValueError):
        return None, None, None, "schema_error", value
    agent = environment.agents[agent_id]
    valid, code, normalized = environment.registry.validate(agent.identity, call)
    if not valid or normalized is None:
        return None, None, None, code, value
    if delegation not in DELEGATION_ACTIONS:
        return None, None, None, "delegation_not_permitted", value
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None, None, None, "confidence_bounds", value
    if not str(value.get("plan_summary", "")).strip():
        return None, None, None, "missing_plan_summary", value
    return normalized, delegation, confidence, "", value


def _repair_prompt(planner: TransformersPlanner, prompt: str, raw: str, error: str) -> str:
    marker = "<|im_start|>assistant\n"
    instruction = (
        "The deterministic validator rejected the JSON (%s). This is the one allowed repair. "
        "Return only one corrected JSON object using the listed action, incident, delegation, and numeric bounds."
        % error[:100]
    )
    if prompt.endswith(marker):
        return (
            prompt + raw[:900] + "<|im_end|>\n<|im_start|>user\n"
            + instruction + "<|im_end|>\n" + marker
        )
    return prompt + "\nREJECTED:\n" + raw[:900] + "\n" + instruction


def _proposal(
    environment: V6PanelEnvironment,
    agent_id: str,
    call: V6ToolCall,
    confidence: float,
) -> V6ActionProposal:
    belief = np.asarray(environment.agents[agent_id].private_beliefs[call.incident_id])
    relevant = [
        index for index, mode in enumerate(INCIDENT_MODES)
        if call.action in (PRIMARY_ACTION_FOR_MODE[mode], SECONDARY_ACTION_FOR_MODE[mode])
    ]
    probability = float(max([belief[index] for index in relevant] or [belief.max()]))
    ordered = np.sort(belief)[::-1]
    agent = environment.agents[agent_id]
    return V6ActionProposal(
        agent_id=agent_id,
        role=agent.identity.role,
        incident_id=call.incident_id,
        action=call.action,
        quantity=call.quantity,
        action_probability=probability,
        action_value=float(confidence),
        value_margin=float(ordered[0] - ordered[1]),
        reason_code=call.reason_code,
    )


def _specifications() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for application in ("humanitarian", "utility_restoration", "commercial"):
        for index in range(QWEN_EPISODE_COUNTS[application]):
            rows.append({
                "application": application,
                "seed": QWEN_BASE_SEEDS[application] + index,
                "regime": QWEN_REGIMES[index % len(QWEN_REGIMES)],
                "information_condition": (
                    "private_fragmented" if index % 2 == 0 else "public_shared"
                ),
            })
    return rows


def _advance_before_decision(environment: V6PanelEnvironment, step: int) -> None:
    environment.current_step = step
    environment._complete_pending(step)
    environment._process_operator_queue(step)
    environment._advance_service(step)
    environment.deliver_observations(step)
    for incident_id in sorted(environment.incidents):
        environment.exchange_sketches(incident_id, step)


def summarize_qwen_decisions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    physical = [value for value in rows if bool(value["accepted_physical_action"])]
    return {
        "decision_epochs": len(rows),
        "physical_action_acceptance": float(len(physical) / max(len(rows), 1)),
        "harmful_action_rate_among_physical": float(np.mean([
            bool(value["harmful"]) for value in physical
        ])) if physical else 0.0,
        "mean_causal_effect_among_physical": float(np.mean([
            float(value["causal_effect"]) for value in physical
        ])) if physical else 0.0,
        "beneficial_physical_actions": sum(bool(value["beneficial"]) for value in physical),
        "neutral_physical_actions": sum(
            abs(float(value["causal_effect"])) <= 1e-12 for value in physical
        ),
        "harmful_physical_actions": sum(bool(value["harmful"]) for value in physical),
    }


def run_real_qwen_qualification(
    repository: Path,
    results_root: Path,
    episode_batch_size: int = 10,
    generation_batch_size: int = 12,
    specifications: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    planner = TransformersPlanner(
        MODEL_IDENTIFIER, MODEL_REVISION,
        max_new_tokens=112, max_input_tokens=2048,
        load_in_4bit=True, seed=66888,
    )
    specs = [dict(value) for value in (specifications or _specifications())]
    all_rows: List[Dict[str, Any]] = []
    episode_summaries: List[Dict[str, Any]] = []
    total_prompt_tokens = total_generated_tokens = total_calls = 0
    total_latency = 0.0
    started = time.perf_counter()
    status_path = results_root / "logs" / "qwen_supervisor_status.json"
    for batch_start in range(0, len(specs), int(episode_batch_size)):
        batch_specs = specs[batch_start:batch_start + int(episode_batch_size)]
        environments: List[Tuple[Dict[str, Any], V6PanelEnvironment, List[Dict[str, Any]]]] = []
        for spec in batch_specs:
            run_id = "v6-qwen-%s-%s-%s-e%d" % (
                spec["application"], spec["regime"],
                spec["information_condition"], int(spec["seed"]),
            )
            episode_path = results_root / "raw" / "qwen" / run_id / "episode.json.gz"
            legacy_path = episode_path.with_suffix("")
            existing_path = episode_path if episode_path.exists() else legacy_path
            if existing_path.exists():
                payload = read_episode_json(existing_path)
                existing_decisions = list(payload["decisions"])
                all_rows.extend(existing_decisions)
                episode_summaries.append(payload["summary"])
                total_calls += sum(int(value.get("llm_calls", 0)) for value in existing_decisions)
                total_prompt_tokens += sum(int(value.get("prompt_tokens", 0)) for value in existing_decisions)
                total_generated_tokens += sum(int(value.get("generated_tokens", 0)) for value in existing_decisions)
                total_latency += float(payload.get("generation_latency_seconds", 0.0))
                continue
            environment = V6PanelEnvironment(
                str(spec["application"]), str(spec["regime"]),
                str(spec["information_condition"]), int(spec["seed"]),
                "event_triggered",
            )
            environments.append(({**spec, "run_id": run_id}, environment, []))
        for step in range(12):
            for _, environment, _ in environments:
                _advance_before_decision(environment, step)
            if step not in V6PanelEnvironment.decision_steps:
                continue
            prompt_cases: List[Dict[str, Any]] = []
            epoch = list(V6PanelEnvironment.decision_steps).index(step)
            for spec, environment, _ in environments:
                incidents = sorted(environment.incidents)
                first = incidents[epoch % len(incidents)]
                second = incidents[(epoch + 1) % len(incidents)]
                agent_ids = [
                    environment.incident_agents[first][0],
                    environment.incident_agents[first][1],
                    environment.incident_agents[second][2],
                ]
                for agent_id in agent_ids:
                    incident_id = environment.agents[agent_id].identity.incident_scope[0]
                    prompt = _prompt(planner, environment, agent_id, incident_id)
                    environment.ledger.append(step, "llm_request", agent_id, {
                        "model_identifier": MODEL_IDENTIFIER,
                        "model_revision": MODEL_REVISION,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "separate_private_context": True,
                        "prompt_text_not_logged": True,
                        "v6": True,
                    }, private_to=agent_id)
                    prompt_cases.append({
                        "spec": spec, "environment": environment,
                        "agent_id": agent_id, "incident_id": incident_id,
                        "step": step, "prompt": prompt,
                    })
            for generation_start in range(0, len(prompt_cases), int(generation_batch_size)):
                cases = prompt_cases[generation_start:generation_start + int(generation_batch_size)]
                prompts = [value["prompt"] for value in cases]
                raw, prompt_counts, generated_counts, latency = planner._generate_prompts(prompts)
                total_latency += float(latency)
                parsed = [
                    _validate(case["environment"], case["agent_id"], response)
                    for case, response in zip(cases, raw)
                ]
                repair_cases = [
                    (index, _repair_prompt(planner, prompts[index], raw[index], parsed[index][3]))
                    for index in range(len(cases)) if parsed[index][0] is None
                ]
                repair_values: Dict[int, Tuple[str, int, int, float]] = {}
                if repair_cases:
                    repair_raw, repair_prompt, repair_generated, repair_latency = planner._generate_prompts(
                        [value[1] for value in repair_cases]
                    )
                    total_latency += float(repair_latency)
                    for (original, _), response, p_count, g_count in zip(
                        repair_cases, repair_raw, repair_prompt, repair_generated,
                    ):
                        repair_values[original] = (
                            response, int(p_count), int(g_count),
                            float(repair_latency / len(repair_cases)),
                        )
                for index, case in enumerate(cases):
                    call, delegation, confidence, error, parsed_value = parsed[index]
                    first_pass = call is not None
                    calls = 1
                    prompt_tokens = int(prompt_counts[index])
                    generated_tokens = int(generated_counts[index])
                    decision_latency = float(latency / max(len(cases), 1))
                    response = raw[index]
                    if call is None:
                        calls += 1
                        response, extra_prompt, extra_generated, repair_case_latency = repair_values[index]
                        prompt_tokens += extra_prompt
                        generated_tokens += extra_generated
                        decision_latency += float(repair_case_latency)
                        call, delegation, confidence, repair_error, parsed_value = _validate(
                            case["environment"], case["agent_id"], response,
                        )
                        if call is None:
                            error = repair_error
                    after_repair = call is not None
                    environment = case["environment"]
                    agent_id = case["agent_id"]
                    incident_id = case["incident_id"]
                    if call is None or delegation is None or confidence is None:
                        call = V6ToolCall(
                            "no_action", incident_id, 0.0,
                            reason_code="one_repair_exhausted",
                        )
                        delegation = "abstain"
                        confidence = 0.0
                    proposal = _proposal(environment, agent_id, call, confidence)
                    context = environment.decision_context(
                        incident_id, case["step"], [proposal], agent_id,
                    )
                    environment.record_candidate(context)
                    environment.ledger.append(case["step"], "llm_structured_response", agent_id, {
                        "proposal": asdict(proposal),
                        "delegation": delegation,
                        "first_pass_valid": first_pass,
                        "valid_after_one_repair": after_repair,
                        "repair_attempted": not first_pass,
                        "validator_error": error,
                        "raw_response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                        "raw_response_not_logged": True,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                        "generation_latency_seconds": decision_latency,
                        "v6": True,
                    }, private_to=agent_id)
                    environment.apply_delegation(context, delegation, "real_qwen_v6")
                    environment.agents[agent_id].vault.remember(
                        agent_id,
                        MemoryRecord(
                            step=case["step"], kind="qwen_plan",
                            summary="%s:%s" % (delegation, call.action),
                            importance=float(confidence),
                        ),
                    )
                    decision = {
                        "run_id": case["spec"]["run_id"],
                        "application": environment.application,
                        "regime": environment.regime,
                        "information_condition": environment.information_condition,
                        "environment_seed": environment.seed,
                        "step": case["step"],
                        "agent_id": agent_id,
                        "role": environment.agents[agent_id].identity.role,
                        "incident_id": incident_id,
                        "private_belief_argmax": int(np.argmax(environment.agents[agent_id].private_beliefs[incident_id])),
                        "selected_action": call.action,
                        "delegation": delegation,
                        "confidence": confidence,
                        "first_pass_valid": first_pass,
                        "repair_attempted": not first_pass,
                        "valid_after_one_repair": after_repair,
                        "llm_calls": calls,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                        "causal_effect": 0.0,
                        "harmful": False,
                        "beneficial": False,
                        "accepted_physical_action": False,
                        "reached_service": False,
                    }
                    next(
                        rows for spec, value, rows in environments
                        if value is environment
                    ).append(decision)
                    total_calls += calls
                    total_prompt_tokens += prompt_tokens
                    total_generated_tokens += generated_tokens
        for spec, environment, decisions in environments:
            episode_calls = sum(int(value.get("llm_calls", 0)) for value in decisions)
            episode_prompt_tokens = sum(int(value.get("prompt_tokens", 0)) for value in decisions)
            episode_generated_tokens = sum(int(value.get("generated_tokens", 0)) for value in decisions)
            summary = environment.finalize("real_qwen_v6")
            candidate_map = {
                (int(value["step"]), str(value["incident_id"]), str(value["agent_id"])): value
                for value in environment.candidate_records
            }
            used_records: set = set()
            for decision in decisions:
                candidate = candidate_map.get((
                    int(decision["step"]), str(decision["incident_id"]),
                    str(decision["agent_id"]),
                ))
                matches = [
                    (index, value) for index, value in enumerate(environment.action_records)
                    if index not in used_records
                    and value["source"] == "autonomous_agent"
                    and value["proposal"]["agent_id"] == decision["agent_id"]
                    and int(value["scheduled_step"]) == int(decision["step"])
                    and value["proposal"]["action"] == decision["selected_action"]
                ]
                if matches:
                    index, value = matches[0]
                    used_records.add(index)
                    decision.update({
                        "causal_effect": float(
                            candidate["evaluator_causal_utility_if_executed"]
                            if candidate is not None else value["causal_effect"]
                        ),
                        "immediate_effect": float(value["causal_effect"]),
                        "harmful": bool(
                            candidate["evaluator_harmful_if_executed"]
                            if candidate is not None else value["harmful"]
                        ),
                        "beneficial": bool(
                            candidate["evaluator_beneficial_if_executed"]
                            if candidate is not None else value["beneficial"]
                        ),
                        "accepted_physical_action": bool(value["accepted_physical_action"]),
                        "reached_service": bool(value["reached_service"]),
                    })
            run_root = results_root / "raw" / "qwen" / spec["run_id"]
            ledger_path = run_root / "events.jsonl.gz"
            ledger_sha = environment.ledger.write_jsonl(ledger_path)
            payload = {
                "study": "Generalized Entropic Consensus V6",
                "stage": "real_qwen_development_qualification",
                "run_id": spec["run_id"],
                "application": environment.application,
                "regime": environment.regime,
                "information_condition": environment.information_condition,
                "environment_seed": environment.seed,
                "model_identifier": MODEL_IDENTIFIER,
                "model_revision": MODEL_REVISION,
                "prompt_revision": PROMPT_REVISION,
                "summary": summary,
                "decisions": decisions,
                "llm_calls": episode_calls,
                "prompt_tokens": episode_prompt_tokens,
                "generated_tokens": episode_generated_tokens,
                # Per-request timings are intentionally not used for inference;
                # this field lets an interrupted run account for completed work.
                "generation_latency_seconds": sum(
                    float(value.get("generation_latency_seconds", 0.0))
                    for value in decisions
                ),
                "event_ledger_path": str(ledger_path.relative_to(results_root)),
                "event_ledger_sha256": ledger_sha,
                "event_ledger_digest": environment.ledger.digest(),
                "simulated_operator": bool(summary["escalations"]),
                "real_human_participants": False,
                "completion_status": "complete",
            }
            episode_path = run_root / "episode.json.gz"
            _atomic_episode(episode_path, payload)
            episode_summaries.append({
                **summary,
                "run_id": spec["run_id"],
                "episode_sha256": sha256_file(episode_path),
                "ledger_sha256": ledger_sha,
            })
            all_rows.extend(decisions)
        atomic_json(status_path, {
            "stage": "qwen_development_qualification",
            "status": "running" if batch_start + episode_batch_size < len(specs) else "complete",
            "completed_episodes": len(episode_summaries),
            "target_episodes": len(specs),
            "updated_at": utc_now(),
        })
    output = results_root / "qwen"
    write_csv(output / "decision_epochs.csv", all_rows)
    write_csv(output / "episode_summary.csv", episode_summaries)
    applications: Dict[str, Any] = {}
    for application in ("commercial", "humanitarian", "utility_restoration"):
        rows = [value for value in all_rows if value["application"] == application]
        physical = [value for value in rows if value["accepted_physical_action"]]
        action_accounting = summarize_qwen_decisions(rows)
        divergent: List[bool] = []
        for run_id in sorted({value["run_id"] for value in rows}):
            episode_rows = [value for value in rows if value["run_id"] == run_id]
            for step in V6PanelEnvironment.decision_steps:
                pair = sorted(
                    [value for value in episode_rows if value["step"] == step],
                    key=lambda value: value["agent_id"],
                )[:2]
                if len(pair) == 2 and pair[0]["private_belief_argmax"] != pair[1]["private_belief_argmax"]:
                    divergent.append(pair[0]["selected_action"] != pair[1]["selected_action"])
        applications[application] = {
            "episodes": len({value["run_id"] for value in rows}),
            **action_accounting,
            "first_pass_validity": float(np.mean([value["first_pass_valid"] for value in rows])),
            "validity_after_one_repair": float(np.mean([value["valid_after_one_repair"] for value in rows])),
            "service_reaching": float(np.mean([value["reached_service"] for value in rows])),
            "action_diversity": len({value["selected_action"] for value in rows}),
            "delegation_diversity": len({value["delegation"] for value in rows}),
            "private_evidence_action_divergence": float(np.mean(divergent)) if divergent else None,
            "communication_requests": sum(value["delegation"] in ("communicate", "request_evidence") for value in rows),
            "abstentions": sum(value["delegation"] == "abstain" for value in rows),
            "human_escalations": sum(value["delegation"] == "escalate_operator" for value in rows),
        }
    report = {
        "study": "Generalized Entropic Consensus V6",
        "stage": "real_qwen_development_qualification",
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "precision": "bitsandbytes NF4 with BF16 computation",
        "episodes": len(episode_summaries),
        "decision_epochs": len(all_rows),
        "applications": applications,
        "llm_calls": total_calls,
        "prompt_tokens": total_prompt_tokens,
        "generated_tokens": total_generated_tokens,
        "generation_latency_seconds": total_latency,
        "wall_seconds_including_model_load": float(time.perf_counter() - started),
        "source_checksum": source_checksum(repository),
        "generated_at": utc_now(),
        "real_qwen_agents": True,
        "simulated_operator": True,
        "real_human_participants": False,
    }
    atomic_json(output / "qualification_summary.json", report)
    return report
