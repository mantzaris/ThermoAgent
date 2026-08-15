"""Substantial real-Qwen qualification of independent V5 agent decisions."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .events import sha256_file
from .planners import TransformersPlanner, extract_json_object
from .v5_environment import V5PanelEnvironment
from .v5_experiments import (
    MODEL_IDENTIFIER, MODEL_REVISION, PROMPT_REVISION, atomic_json,
    git_metadata, protocol_checksum, source_checksum, utc_now, write_csv,
)
from .v5_tools import V5ToolCall, V5ToolRegistry


QUALIFICATION_BASE_SEEDS = (51901, 51902, 51903, 51904)
REGIMES = ("telemetry_integrity", "partition", "compound")


def _prompt(
    planner: TransformersPlanner,
    environment: V5PanelEnvironment,
    agent_id: str,
) -> str:
    agent = environment.agents[agent_id]
    context = agent.context()
    registry = V5ToolRegistry()
    payload = {
        "identity": context["identity"],
        "private_observation": context["private_observation"],
        "private_belief_for_scoped_incident": context["private_beliefs"],
        "private_utility": context["utility"],
        "private_memory": context["private_memory"],
        "explicitly_delivered_messages": context["inbox"],
        "private_commitments": context["commitments"],
        "typed_tool_schema": registry.schema(agent.identity.role),
        "allowed_actions": list(registry.allowed_actions(agent.identity.role)),
        "incident_scope": list(agent.identity.incident_scope),
    }
    system = (
        "You are one independent autonomous organization in an abstract logistics or defensive utility-restoration simulation. "
        "You cannot see another agent's private observation, evaluator state, true incident mode, future event, or counterfactual. "
        "Choose one bounded action using only the supplied private context. Return exactly one compact JSON object with keys "
        "action, incident_id, quantity, reason_code, plan_summary, justification, confidence. "
        "The action must be in allowed_actions; incident_id must be in incident_scope; quantity must be 0 through 2. "
        "Keep reason_code, plan_summary, and justification under 12 words. Do not use markdown or reveal hidden reasoning."
    )
    user = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return planner.tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def _validate(
    environment: V5PanelEnvironment,
    agent_id: str,
    raw: str,
) -> Tuple[Optional[V5ToolCall], str, Dict[str, Any]]:
    value = extract_json_object(raw)
    if value is None:
        return None, "no_json_object", {}
    if not isinstance(value, dict):
        return None, "response_not_object", {}
    tool_payload = {
        key: value[key]
        for key in ("action", "incident_id", "quantity", "target_agent", "reason_code")
        if key in value
    }
    agent = environment.agents[agent_id]
    validation = V5ToolRegistry().validate(
        agent.identity.role, agent.identity.incident_scope, tool_payload,
    )
    if not validation.ok:
        return None, validation.code, value
    for required in ("plan_summary", "justification", "confidence"):
        if required not in value:
            return None, "missing_%s" % required, value
    try:
        confidence = float(value["confidence"])
    except (TypeError, ValueError):
        return None, "confidence_type", value
    if not 0.0 <= confidence <= 1.0:
        return None, "confidence_bounds", value
    return validation.normalized, "", value


def _repair_prompt(planner: TransformersPlanner, prompt: str, raw: str, error: str) -> str:
    marker = "<|im_start|>assistant\n"
    instruction = (
        "The deterministic validator rejected the JSON (%s). This is the one allowed repair. "
        "Return only one corrected JSON object using an allowed action and scoped incident."
        % error[:100]
    )
    if prompt.endswith(marker):
        return prompt + raw[:900] + "<|im_end|>\n<|im_start|>user\n" + instruction + "<|im_end|>\n" + marker
    return prompt + "\nREJECTED:\n" + raw[:900] + "\n" + instruction


def _write_qwen_episode(
    repository: Path,
    results_root: Path,
    episode: Mapping[str, Any],
    environment: V5PanelEnvironment,
) -> None:
    run_id = str(episode["run_id"])
    run_root = results_root / "raw" / "real_qwen_qualification" / run_id
    episode_path = run_root / "episode.json"
    ledger_path = run_root / "events.jsonl.gz"
    manifest_path = results_root / "manifests" / "real_qwen_qualification" / (run_id + ".json")
    if episode_path.exists() or ledger_path.exists() or manifest_path.exists():
        raise FileExistsError("Qwen V5 qualification output exists: %s" % run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    ledger_sha = environment.ledger.write_jsonl(ledger_path)
    atomic_json(episode_path, dict(episode))
    git = git_metadata(repository)
    atomic_json(manifest_path, {
        "run_id": run_id,
        "stage": "real_qwen_qualification",
        "application": episode["application"],
        "regime": episode["regime"],
        "environment_seed": episode["environment_seed"],
        "source_checksum": source_checksum(repository),
        "protocol_checksum": protocol_checksum(repository),
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "dirty_tree": git["dirty"],
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "prompt_revision": PROMPT_REVISION,
        "precision": "bitsandbytes NF4 with BF16 computation",
        "planner": "real_Qwen_independent_agent",
        "simulated_operator": False,
        "real_human_participants": False,
        "event_count": len(environment.ledger.events),
        "event_ledger_digest": environment.ledger.digest(),
        "episode_sha256": sha256_file(episode_path),
        "ledger_sha256": ledger_sha,
        "completion_status": "complete",
        "generated_at": utc_now(),
    })


def run_real_qwen_qualification(
    repository: Path,
    results_root: Path,
    base_seeds: Sequence[int] = QUALIFICATION_BASE_SEEDS,
    batch_size: int = 12,
) -> Dict[str, Any]:
    planner = TransformersPlanner(
        MODEL_IDENTIFIER, MODEL_REVISION,
        max_new_tokens=128, max_input_tokens=2304,
        load_in_4bit=True, seed=51999,
    )
    episodes: List[Dict[str, Any]] = []
    cases: List[Dict[str, Any]] = []
    for application in ("commercial", "humanitarian", "utility_restoration"):
        for base_seed in base_seeds:
            for regime_index, regime in enumerate(REGIMES):
                environment_seed = int(base_seed + 100 * regime_index)
                environment = V5PanelEnvironment(
                    application, regime, "private_fragmented", environment_seed,
                    sketch_policy="event_triggered",
                )
                run_id = "v5-qwen-%s-%s-e%d" % (application, regime, environment_seed)
                episode = {
                    "run_id": run_id,
                    "application": application,
                    "regime": regime,
                    "environment_seed": environment_seed,
                    "environment": environment,
                    "decisions": [],
                }
                # Two agents see distinct private evidence for one incident; a
                # third agent acts on another incident. This tests divergence,
                # not just one canned action.
                first_incident = list(environment.incidents)[regime_index % len(environment.incidents)]
                second_incident = list(environment.incidents)[(regime_index + 1) % len(environment.incidents)]
                selected_agents = [
                    environment.incident_agents[first_incident][0],
                    environment.incident_agents[first_incident][1],
                    environment.incident_agents[second_incident][2],
                ]
                for epoch, agent_id in enumerate(selected_agents):
                    prompt = _prompt(planner, environment, agent_id)
                    environment.ledger.append(
                        environment.incidents[environment.agents[agent_id].identity.incident_scope[0]].disruption_step + epoch,
                        "llm_request", agent_id,
                        {
                            "model_identifier": MODEL_IDENTIFIER,
                            "model_revision": MODEL_REVISION,
                            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                            "separate_agent_context": True,
                            "prompt_text_not_logged": True,
                            "v5": True,
                        }, private_to=agent_id,
                    )
                    cases.append({
                        "episode": episode,
                        "environment": environment,
                        "agent_id": agent_id,
                        "epoch": epoch,
                        "prompt": prompt,
                    })
                episodes.append(episode)
    started = time.perf_counter()
    for start in range(0, len(cases), int(batch_size)):
        batch = cases[start:start + int(batch_size)]
        prompts = [case["prompt"] for case in batch]
        raw_values, prompt_tokens, generated_tokens, latency = planner._generate_prompts(prompts)
        repair_cases: List[Tuple[int, str]] = []
        parsed: List[Tuple[Optional[V5ToolCall], str, Dict[str, Any]]] = []
        for index, case in enumerate(batch):
            result = _validate(case["environment"], case["agent_id"], raw_values[index])
            parsed.append(result)
            if result[0] is None:
                repair_cases.append((index, _repair_prompt(planner, prompts[index], raw_values[index], result[1])))
        repair_outputs: Dict[int, Tuple[str, int, int, float]] = {}
        if repair_cases:
            repair_raw, repair_prompt_tokens, repair_generated_tokens, repair_latency = planner._generate_prompts([value[1] for value in repair_cases])
            for repair_index, ((original_index, _), raw, p_tokens, g_tokens) in enumerate(zip(repair_cases, repair_raw, repair_prompt_tokens, repair_generated_tokens)):
                repair_outputs[original_index] = (raw, int(p_tokens), int(g_tokens), float(repair_latency / len(repair_cases)))
        for index, case in enumerate(batch):
            tool_call, error, value = parsed[index]
            first_pass = tool_call is not None
            calls = 1
            total_prompt = int(prompt_tokens[index])
            total_generated = int(generated_tokens[index])
            total_latency = float(latency / len(batch))
            final_raw = raw_values[index]
            if tool_call is None:
                calls += 1
                final_raw, added_prompt, added_generated, added_latency = repair_outputs[index]
                total_prompt += added_prompt
                total_generated += added_generated
                total_latency += added_latency
                tool_call, repair_error, value = _validate(case["environment"], case["agent_id"], final_raw)
                error = repair_error if tool_call is None else error
            valid_after_repair = tool_call is not None
            if tool_call is None:
                agent = case["environment"].agents[case["agent_id"]]
                tool_call = V5ToolCall("no_action", agent.identity.incident_scope[0], 0.0, None, "repair_exhausted")
            environment = case["environment"]
            effect = environment.action_effect(tool_call.incident_id, tool_call.action)
            agent = environment.agents[case["agent_id"]]
            environment.ledger.append(
                environment.incidents[tool_call.incident_id].disruption_step + int(case["epoch"]),
                "llm_structured_response", case["agent_id"],
                {
                    "tool_call": tool_call.as_dict(),
                    "first_pass_valid": first_pass,
                    "valid_after_one_repair": valid_after_repair,
                    "repair_attempted": not first_pass,
                    "validator_error": error,
                    "raw_response_sha256": hashlib.sha256(final_raw.encode("utf-8")).hexdigest(),
                    "raw_response_not_logged": True,
                    "prompt_tokens": total_prompt,
                    "generated_tokens": total_generated,
                    "v5": True,
                }, private_to=case["agent_id"],
            )
            environment.ledger.append(
                environment.incidents[tool_call.incident_id].disruption_step + int(case["epoch"]),
                "tool_call", case["agent_id"],
                {"tool_call": tool_call.as_dict(), "role": agent.identity.role, "v5": True},
            )
            environment.ledger.append(
                environment.incidents[tool_call.incident_id].disruption_step + int(case["epoch"]),
                "tool_result", "simulator",
                {
                    "agent_id": case["agent_id"],
                    "accepted": effect.accepted_action,
                    "reached_next_stage": effect.reached_next_stage,
                    "reached_service": effect.reached_service,
                    "causal_effect": effect.causal_effect,
                    "v5": True,
                }, private_to=case["agent_id"],
            )
            decision = {
                "agent_id": case["agent_id"],
                "role": agent.identity.role,
                "incident_id": tool_call.incident_id,
                "epoch": int(case["epoch"]),
                "private_belief_argmax": int(max(range(len(agent.private_beliefs[tool_call.incident_id])), key=lambda value_index: agent.private_beliefs[tool_call.incident_id][value_index])),
                "selected_action": tool_call.action,
                "first_pass_valid": first_pass,
                "repair_attempted": not first_pass,
                "valid_after_one_repair": valid_after_repair,
                "material_action_accepted": effect.accepted_action,
                "reached_next_stage": effect.reached_next_stage,
                "reached_service": effect.reached_service,
                "causal_effect": effect.causal_effect,
                "llm_calls": calls,
                "prompt_tokens": total_prompt,
                "generated_tokens": total_generated,
                "latency_seconds": total_latency,
            }
            case["episode"]["decisions"].append(decision)
    flat_rows: List[Dict[str, Any]] = []
    for episode in episodes:
        environment = episode.pop("environment")
        decisions = episode["decisions"]
        payload = {
            **episode,
            "study": "ThermoHITL v5",
            "stage": "real_qwen_qualification",
            "model_identifier": MODEL_IDENTIFIER,
            "model_revision": MODEL_REVISION,
            "decisions": decisions,
            "event_ledger_digest": environment.ledger.digest(),
            "stochastic_tape_digest": environment.stochastic_tape_digest,
            "simulated_operator": False,
            "real_human_participants": False,
        }
        _write_qwen_episode(repository, results_root, payload, environment)
        for row in decisions:
            flat_rows.append({"run_id": episode["run_id"], "application": episode["application"], "regime": episode["regime"], "environment_seed": episode["environment_seed"], **row})
    write_csv(results_root / "development" / "real_qwen_qualification" / "decision_epochs.csv", flat_rows)
    by_application: Dict[str, Any] = {}
    for application in ("commercial", "humanitarian", "utility_restoration"):
        rows = [row for row in flat_rows if row["application"] == application]
        episode_ids = {row["run_id"] for row in rows}
        paired_divergence: List[bool] = []
        for run_id in episode_ids:
            pair = sorted([row for row in rows if row["run_id"] == run_id], key=lambda item: item["epoch"])[:2]
            if len(pair) == 2 and pair[0]["private_belief_argmax"] != pair[1]["private_belief_argmax"]:
                paired_divergence.append(pair[0]["selected_action"] != pair[1]["selected_action"])
        by_application[application] = {
            "episodes": len(episode_ids),
            "decision_epochs": len(rows),
            "first_pass_validity": float(sum(row["first_pass_valid"] for row in rows) / len(rows)),
            "validity_after_one_repair": float(sum(row["valid_after_one_repair"] for row in rows) / len(rows)),
            "material_acceptance": float(sum(row["material_action_accepted"] for row in rows) / len(rows)),
            "service_reaching": float(sum(row["reached_service"] for row in rows) / len(rows)),
            "action_diversity": len({row["selected_action"] for row in rows}),
            "private_evidence_action_divergence": float(sum(paired_divergence) / len(paired_divergence)) if paired_divergence else None,
        }
    report = {
        "study": "ThermoHITL v5",
        "stage": "real_qwen_qualification",
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "precision": "bitsandbytes NF4 with BF16 computation",
        "episodes": len(episodes),
        "decision_epochs": len(flat_rows),
        "applications": by_application,
        "llm_calls": sum(row["llm_calls"] for row in flat_rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in flat_rows),
        "generated_tokens": sum(row["generated_tokens"] for row in flat_rows),
        "latency_seconds": sum(row["latency_seconds"] for row in flat_rows),
        "wall_seconds_including_model_load": float(time.perf_counter() - started),
        "real_qwen_agents": True,
        "simulated_operator": False,
        "real_human_participants": False,
    }
    atomic_json(results_root / "development" / "real_qwen_qualification.json", report)
    return report
