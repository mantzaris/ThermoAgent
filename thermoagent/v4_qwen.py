"""Real-Qwen structured-action qualification for all three v4 applications."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .planners import TransformersPlanner, coerce_plan, extract_json_object
from .types import PlanOutput
from .v4_environment import FragmentedOversightEnvironment
from .v4_experiments import (
    MODEL_IDENTIFIER,
    MODEL_REVISION,
    write_episode,
)
from .v4_runner import V4EpisodeResult


QUALIFICATION_SEEDS = (24901, 24902)
STAGE = "development_real_qwen_qualification"


def _agent_and_action(
    environment: FragmentedOversightEnvironment,
    seed_index: int,
) -> Tuple[str, str, Dict[str, Any]]:
    if environment.application == "commercial":
        agent = next(value for value in environment.agents.values() if value.identity.role == "supplier")
        incident = agent.vault.observation(agent.agent_id).incident_id
        return agent.agent_id, "schedule_shipment", {
            "target": incident, "quantity": 1.0,
            "arrival_step": environment.step_index + 2,
        }
    if environment.application == "humanitarian":
        agent = next(value for value in environment.agents.values() if value.identity.role == "ngo")
        incident = agent.vault.observation(agent.agent_id).incident_id
        return agent.agent_id, "transfer_resource", {
            "target": incident, "quantity": 1.0,
            "arrival_step": environment.step_index + 2,
        }
    role = "crew_dispatch" if seed_index == 0 else "parts_depot"
    agent = next(value for value in environment.agents.values() if value.identity.role == role)
    incident = agent.vault.observation(agent.agent_id).incident_id
    if role == "crew_dispatch":
        return agent.agent_id, "dispatch_field_crew", {
            "crew_id": "crew_1", "target_zone": incident, "skill": "electrical",
        }
    return agent.agent_id, "allocate_spare_component", {
        "component": environment.incidents[incident].resource_required,
        "quantity": 1, "target_zone": incident,
    }


def _prompt(
    planner: TransformersPlanner,
    environment: FragmentedOversightEnvironment,
    agent_id: str,
    tool: str,
    exact_arguments: Mapping[str, Any],
) -> str:
    agent = environment.agents[agent_id]
    context = agent.context(environment.ledger)
    schema = environment.registry.prompt_schema(agent.identity.role, {tool})[tool]
    payload = {
        "identity": context["identity"],
        "private_observation": context["observation"],
        "private_utility": context["utility"],
        "private_memory": context["episodic_memory"],
        "delivered_messages": context["messages"],
        "private_commitments": context["commitments"],
        "selected_coordination_option": "execute_one_bounded_material_response",
        "exact_allowed_tool": tool,
        "typed_schema": schema,
        "exact_permitted_arguments": dict(exact_arguments),
    }
    system = (
        "You are one independent autonomous logistics organization in an abstract defensive simulation. "
        "Use only your private observation, memory, utility, commitments, and explicitly delivered messages. "
        "Return exactly one compact JSON object with keys plan_summary, tool, arguments, justification, confidence. "
        "Use the exact_allowed_tool and copy exact_permitted_arguments exactly. Do not use markdown or reveal chain-of-thought. "
        "Keep plan_summary and justification under 12 words each."
    )
    user = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return planner.tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _validate(
    environment: FragmentedOversightEnvironment,
    agent_id: str,
    expected_tool: str,
    expected_arguments: Mapping[str, Any],
    raw: str,
) -> Tuple[Optional[PlanOutput], str]:
    value = extract_json_object(raw)
    if value is None:
        return None, "no_json_object"
    try:
        plan = coerce_plan(value)
    except (TypeError, ValueError) as error:
        return None, "schema:%s" % str(error)[:100]
    validation = environment.registry.validate(
        environment.agents[agent_id].identity.role, plan
    )
    if not validation.ok:
        return None, "typed_schema:%s" % validation.code
    if plan.tool != expected_tool:
        return None, "wrong_tool:%s" % plan.tool
    if plan.arguments != dict(expected_arguments):
        return None, "arguments_do_not_match_bounded_affordance"
    return plan, ""


def _repair_prompt(planner: TransformersPlanner, prompt: str, raw: str, error: str) -> str:
    marker = "<|im_start|>assistant\n"
    instruction = (
        "The deterministic validator rejected the response (%s). This is the single permitted repair. "
        "Return one JSON object only, copying exact_allowed_tool and exact_permitted_arguments exactly."
        % error[:120]
    )
    if prompt.endswith(marker):
        return (
            prompt + raw[:1000] + "<|im_end|>\n<|im_start|>user\n"
            + instruction + "<|im_end|>\n" + marker
        )
    return prompt + "\nREJECTED:\n" + raw[:1000] + "\n" + instruction


def run_real_qwen_qualification(
    repository: Path,
    results_root: Path,
    seeds: Sequence[int] = QUALIFICATION_SEEDS,
) -> Dict[str, Any]:
    planner = TransformersPlanner(
        MODEL_IDENTIFIER,
        MODEL_REVISION,
        max_new_tokens=128,
        max_input_tokens=2560,
        load_in_4bit=True,
        seed=24999,
    )
    cases: List[Dict[str, Any]] = []
    for application in ("commercial", "humanitarian", "utility_restoration"):
        pending: List[Dict[str, Any]] = []
        prompts: List[str] = []
        for seed_index, seed in enumerate(seeds):
            environment = FragmentedOversightEnvironment(
                application=application,
                regime="compound",
                information_condition="globally_public",
                seed=int(seed),
                horizon=20,
                disruption_step=6,
                communication_enabled=True,
            )
            for _ in range(8):
                environment.step()
                environment.deliver_observations()
                environment.exchange_sketches(gossip_rounds=3)
            agent_id, tool, arguments = _agent_and_action(environment, seed_index)
            prompt = _prompt(planner, environment, agent_id, tool, arguments)
            environment.ledger.append(
                environment.step_index,
                "llm_request",
                agent_id,
                {
                    "model_identifier": MODEL_IDENTIFIER,
                    "model_revision": MODEL_REVISION,
                    "prompt_sha256": __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest(),
                    "separate_agent_context": True,
                    "prompt_text_not_logged": True,
                },
                private_to=agent_id,
            )
            prompts.append(prompt)
            pending.append({
                "environment": environment,
                "seed": int(seed),
                "agent_id": agent_id,
                "tool": tool,
                "arguments": arguments,
                "prompt": prompt,
            })
        raw_values, prompt_tokens, generated_tokens, latency = planner._generate_prompts(prompts)
        for index, case in enumerate(pending):
            environment = case["environment"]
            plan, error = _validate(
                environment, case["agent_id"], case["tool"],
                case["arguments"], raw_values[index],
            )
            first_pass = plan is not None
            repair_attempted = not first_pass
            total_prompt_tokens = int(prompt_tokens[index])
            total_generated_tokens = int(generated_tokens[index])
            total_latency = float(latency / max(len(pending), 1))
            final_raw = raw_values[index]
            if plan is None:
                repair_prompt = _repair_prompt(planner, case["prompt"], raw_values[index], error)
                repair_raw, repair_prompt_tokens, repair_generated_tokens, repair_latency = planner._generate_prompts([repair_prompt])
                final_raw = repair_raw[0]
                total_prompt_tokens += int(repair_prompt_tokens[0])
                total_generated_tokens += int(repair_generated_tokens[0])
                total_latency += float(repair_latency)
                plan, repair_error = _validate(
                    environment, case["agent_id"], case["tool"],
                    case["arguments"], final_raw,
                )
                error = repair_error if plan is None else error
            valid_after_repair = plan is not None
            if plan is None:
                plan = PlanOutput(
                    "Invalid output; abstain safely.", "no_op", {},
                    "The single repair was exhausted.", 0.0,
                )
            environment.ledger.append(
                environment.step_index,
                "llm_structured_response",
                case["agent_id"],
                {
                    "plan": plan.as_dict(),
                    "first_pass_valid": first_pass,
                    "repair_attempted": repair_attempted,
                    "valid_after_one_repair": valid_after_repair,
                    "validator_error": error,
                    "raw_response_sha256": __import__("hashlib").sha256(final_raw.encode("utf-8")).hexdigest(),
                    "raw_response_not_logged": True,
                    "prompt_tokens": total_prompt_tokens,
                    "generated_tokens": total_generated_tokens,
                },
                private_to=case["agent_id"],
            )
            before = dict(environment.metric_counters)
            result = environment.validate_and_execute_plan(case["agent_id"], plan)
            for _ in range(3):
                environment.step()
                environment.deliver_observations()
                environment.exchange_sketches(gossip_rounds=3)
            accepted = int(
                environment.metric_counters["material_actions_accepted"]
                - before["material_actions_accepted"]
            )
            next_stage = int(
                environment.metric_counters["material_actions_next_stage"]
                - before["material_actions_next_stage"]
            )
            reached_service = int(
                environment.metric_counters["material_actions_reached_service"]
                - before["material_actions_reached_service"]
            )
            transitions = [
                event for event in environment.ledger.events
                if event.kind == "v4_state_transition"
            ]
            time_series = [
                {"step": int(event.step), "loss": float(event.payload["loss"])}
                for event in transitions
            ]
            conservation = environment.conservation_report()
            run_id = "%s-%s-e%d" % (STAGE, application, case["seed"])
            metrics = {
                "primary_outcome": float(sum(value["loss"] for value in time_series)),
                "structured_attempts": 1,
                "first_pass_valid": int(first_pass),
                "valid_after_repair": int(valid_after_repair),
                "material_actions_accepted": accepted,
                "material_actions_next_stage": next_stage,
                "material_actions_reached_service": reached_service,
                "llm_calls": 1 + int(repair_attempted),
                "prompt_tokens": total_prompt_tokens,
                "generated_tokens": total_generated_tokens,
                "llm_latency_seconds": total_latency,
                "maximum_conservation_residual": conservation["maximum_residual"],
                "conservation_feasible": conservation["feasible"],
            }
            episode = V4EpisodeResult(
                run_id=run_id,
                application=application,
                regime="compound",
                information_condition="globally_public",
                method="real_qwen_actionability_qualification",
                environment_seed=case["seed"],
                operator_seed=0,
                rl_seed=None,
                status="complete" if conservation["feasible"] else "failed",
                metrics=metrics,
                time_series=time_series,
                candidate_interventions=[],
                counterfactuals=[],
                manifest_fields={
                    "run_id": run_id,
                    "stage": STAGE,
                    "application": application,
                    "regime": "compound",
                    "information_condition": "globally_public",
                    "method": "real_qwen_actionability_qualification",
                    "environment_seed": case["seed"],
                    "operator_seed": 0,
                    "planner_seed": 24999,
                    "rl_seed": None,
                    "completion_status": "complete" if conservation["feasible"] else "failed",
                    "failure_reason": None if conservation["feasible"] else "conservation_or_feasibility",
                    "event_count": len(environment.ledger.events),
                    "event_ledger_digest": environment.ledger.digest(),
                    "simulated_operator": False,
                    "planner": "real_qwen_transformers_nf4",
                    "model_identifier": MODEL_IDENTIFIER,
                    "model_revision": MODEL_REVISION,
                    "precision": "bitsandbytes NF4; BF16 computation",
                    "serving_library": "Transformers 4.55.4",
                    "llm_calls": metrics["llm_calls"],
                    "prompt_tokens": total_prompt_tokens,
                    "generated_tokens": total_generated_tokens,
                    "llm_latency_seconds": total_latency,
                    "single_gpu_hours": total_latency / 3600.0,
                },
                ledger=environment.ledger,
            )
            write_episode(repository, results_root, STAGE, episode)
            cases.append({
                "run_id": run_id,
                "application": application,
                "environment_seed": case["seed"],
                "agent_id": case["agent_id"],
                "role": environment.agents[case["agent_id"]].identity.role,
                "expected_tool": case["tool"],
                "executed_tool": plan.tool,
                "first_pass_valid": first_pass,
                "repair_attempted": repair_attempted,
                "valid_after_one_repair": valid_after_repair,
                "material_action_accepted": accepted,
                "material_action_next_stage": next_stage,
                "material_action_reached_service": reached_service,
                "tool_result_ok": result.ok,
                "llm_calls": metrics["llm_calls"],
                "prompt_tokens": total_prompt_tokens,
                "generated_tokens": total_generated_tokens,
                "latency_seconds": total_latency,
                "raw_output_retained": False,
            })
    by_application: Dict[str, Any] = {}
    for application in ("commercial", "humanitarian", "utility_restoration"):
        rows = [value for value in cases if value["application"] == application]
        attempts = len(rows)
        summary = {
            "episodes": attempts,
            "first_pass_validity": sum(value["first_pass_valid"] for value in rows) / max(attempts, 1),
            "validity_after_one_repair": sum(value["valid_after_one_repair"] for value in rows) / max(attempts, 1),
            "accepted_to_next_stage": sum(value["material_action_next_stage"] for value in rows) / max(sum(value["material_action_accepted"] for value in rows), 1),
            "accepted_to_service": sum(value["material_action_reached_service"] for value in rows) / max(sum(value["material_action_accepted"] for value in rows), 1),
        }
        summary["passed"] = bool(
            summary["first_pass_validity"] >= 0.90
            and summary["validity_after_one_repair"] >= 0.98
            and summary["accepted_to_next_stage"] >= 0.70
            and summary["accepted_to_service"] >= 0.30
        )
        by_application[application] = summary
    report = {
        "study": "ThermoHITL v4",
        "stage": STAGE,
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "precision": "bitsandbytes NF4; BF16 computation",
        "simulated_environment": True,
        "real_qwen_agents": True,
        "real_human_operators": False,
        "episodes": len(cases),
        "cases": cases,
        "applications": by_application,
        "passed": all(value["passed"] for value in by_application.values()),
        "llm_calls": sum(value["llm_calls"] for value in cases),
        "prompt_tokens": sum(value["prompt_tokens"] for value in cases),
        "generated_tokens": sum(value["generated_tokens"] for value in cases),
        "llm_latency_seconds": sum(value["latency_seconds"] for value in cases),
    }
    destination = results_root / "development" / "real_qwen_qualification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
