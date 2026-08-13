"""Real-model CUDA and independent-agent smoke experiments."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .environment import LogisticsEnvironment, ScenarioConfig
from .planners import (
    PLANNER_PROMPT_REVISION,
    PlannerRequest,
    TransformersPlanner,
    validate_request_plan,
)
from .runner import EpisodeRunner
from .tools import ToolRegistry
from .types import Commitment, CoordinationOption, Message


def model_smoke(
    output: Path,
    model_id: str,
    revision: str,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    import torch
    import transformers
    import bitsandbytes

    started_at = datetime.now(timezone.utc).isoformat()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    planner = TransformersPlanner(model_id, revision, max_new_tokens=max_new_tokens, load_in_4bit=True)
    load_seconds = time.perf_counter() - load_started
    env = LogisticsEnvironment(ScenarioConfig(application="commercial", seed=900, horizon=4, n_agents=5, disruption="moderate"))
    env.transition()
    env.deliver_observations()
    identities = env.public_identities()
    requests: List[PlannerRequest] = []
    for index, agent_id in enumerate(env.agent_ids[:2]):
        agent = env.agents[agent_id]
        requests.append(PlannerRequest(
            agent_id=agent_id,
            role=agent.identity.role,
            application="commercial",
            option=1 if index == 0 else 3,
            context=agent.retrieve_context(env.step_index, env.ledger),
            candidate_agents=identities,
        ))
    inference_started = time.perf_counter()
    responses = planner.plan_batch(requests)
    inference_seconds = time.perf_counter() - inference_started
    registry = ToolRegistry()
    validations = [registry.validate(request.role, response.output) for request, response in zip(requests, responses)]
    executions = [
        env.execute_tool(request.agent_id, response.output.tool, validation.data) if validation.ok else validation
        for request, response, validation in zip(requests, responses, validations)
    ]
    matrix = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    product = matrix @ matrix
    torch.cuda.synchronize()
    record = {
        "status": "complete",
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "model_identifier": model_id,
        "model_revision": revision,
        "prompt_template_revision": PLANNER_PROMPT_REVISION,
        "precision": "bitsandbytes NF4; bfloat16 compute; double quantization",
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds": load_seconds,
        "batched_inference_seconds": inference_seconds,
        "batch_size": len(requests),
        "prompt_tokens": sum(response.prompt_tokens for response in responses),
        "generated_tokens": sum(response.generated_tokens for response in responses),
        "generated_tokens_per_second": sum(response.generated_tokens for response in responses) / max(inference_seconds, 1e-9),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "valid_json_rate": sum(response.valid_json for response in responses) / len(responses),
        "static_schema_valid_rate": sum(validation.ok for validation in validations) / len(validations),
        "valid_tool_rate": sum(execution.ok for execution in executions) / len(executions),
        "responses": [
            {
                "agent_id": request.agent_id,
                "role": request.role,
                "option": request.option,
                "plan": response.output.as_dict(),
                "valid_json": response.valid_json,
                "recovery": response.recovery,
                "validation": validation.as_dict(),
                "execution": execution.as_dict(),
                "prompt_tokens": response.prompt_tokens,
                "generated_tokens": response.generated_tokens,
            }
            for request, response, validation, execution in zip(requests, responses, validations, executions)
        ],
        "cuda_matrix_finite": bool(torch.isfinite(product).all().item()),
        "independent_contexts": len({request.agent_id for request in requests}) == len(requests),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not record["cuda_matrix_finite"] or record["valid_json_rate"] == 0:
        raise RuntimeError("real-model smoke validation failed")
    return record


def agentic_smoke(output_dir: Path, model_id: str, revision: str) -> Dict[str, Any]:
    """Stage 1 real-LLM negotiation plus both short applications."""
    planner = TransformersPlanner(model_id, revision, max_new_tokens=160, load_in_4bit=True)
    registry = ToolRegistry()
    env = LogisticsEnvironment(ScenarioConfig(
        application="commercial", seed=902, horizon=8, n_agents=11,
        private_information=1.0, objective_misalignment=1.0,
        communication="reliable", disruption="moderate", decision_interval=2,
    ))
    env.transition()
    env.deliver_observations()
    seller = next(a for a in env.agent_ids if env.agents[a].identity.role == "supplier")
    buyers = [a for a in env.agent_ids if env.agents[a].identity.role == "retailer"]
    records: List[Dict[str, Any]] = []

    def ask(
        target_env: LogisticsEnvironment,
        agent_id: str,
        option: int,
        label: str,
        candidate_agents: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        agent = target_env.agents[agent_id]
        request = PlannerRequest(
            agent_id,
            agent.identity.role,
            target_env.config.application,
            option,
            agent.retrieve_context(target_env.step_index, target_env.ledger),
            candidate_agents if candidate_agents is not None else target_env.public_identities(),
        )
        response = planner.plan_batch([request])[0]
        validation = registry.validate(agent.identity.role, response.output)
        affordance_error = validate_request_plan(request, response.output)
        if validation.ok and affordance_error is not None:
            validation = affordance_error
        result = target_env.execute_tool(agent_id, response.output.tool, validation.data) if validation.ok else validation
        if not validation.ok:
            target_env.ledger.append(
                target_env.step_index,
                "tool_result",
                agent_id,
                {"tool": response.output.tool, **result.as_dict()},
                private_to=agent_id,
            )
        agent.reflect(target_env.step_index, response.output.plan_summary, result.ok, result.code)
        records.append({
            "label": label, "agent_id": agent_id, "option": option,
            "plan": response.output.as_dict(), "valid_json": response.valid_json,
            "validation": validation.as_dict(), "tool_result": result.as_dict(),
            "prompt_tokens": response.prompt_tokens, "generated_tokens": response.generated_tokens,
            "latency_seconds": response.latency_seconds,
        })
        return result

    def ask_for_code(
        target_env: LogisticsEnvironment,
        agent_id: str,
        option: int,
        label: str,
        expected_code: str,
        candidate_agents: Optional[List[Dict[str, Any]]] = None,
        attempts: int = 3,
    ) -> Any:
        """Use a fixed retry budget and retain every invalid proposal."""
        result = None
        for attempt in range(1, attempts + 1):
            result = ask(
                target_env,
                agent_id,
                option,
                "%s (attempt %d/%d)" % (label, attempt, attempts),
                candidate_agents,
            )
            if result.code == expected_code:
                break
        return result

    # An explicit logged quote request gives the seller a legitimate reason to
    # use private cost/inventory to initiate one actual contract.
    quote_message = Message(
        message_id="M_STAGE1_QUOTE", sender=buyers[0], recipient=seller,
        kind="quote_request", payload={"quantity": 6.0, "due_step": env.step_index + 4},
        sent_step=env.step_index, deliver_step=env.step_index,
    )
    env.ledger.append(env.step_index, "message", buyers[0], asdict(quote_message), private_to=seller)
    env.agents[seller].deliver_message(quote_message)
    ask_for_code(env, seller, 3, "private-cost seller offer", "offer_submitted")

    # Two distinct buyers exercise independent authority under different private
    # reservation values. The fixtures are logged offers, not hidden decisions.
    response_specs = [
        (buyers[0], 0.45, 4.5, "reject strongly irrational offer"),
        (buyers[1], 1.00, 1.20, "counter slightly expensive offer"),
    ]
    for buyer, reservation, price, label in response_specs:
        env.agents[buyer].utility.reservation_price = reservation
        commitment = Commitment(
            commitment_id=env._next_id("commitment"), proposer=seller, partner=buyer,
            quantity=6.0, unit_price=price, due_step=env.step_index + 4,
        )
        env.commitments[commitment.commitment_id] = commitment
        env.agents[seller].commitments[commitment.commitment_id] = Commitment(**commitment.__dict__)
        env.agents[buyer].commitments[commitment.commitment_id] = Commitment(**commitment.__dict__)
        env.ledger.append(env.step_index, "offer", seller, asdict(commitment), private_to=buyer)
        env.agents[buyer].last_plan_summary = (
            "Evaluate the pending offer against the strict private reservation price; "
            "counter only when a feasible reservation-price agreement remains."
        )
        expected = "offer_rejected" if reservation < 0.5 else "offer_countered"
        ask_for_code(env, buyer, 4, label, expected)

    # A stressed warehouse explicitly considers a temporary coalition.
    coalition_agent = next(a for a in env.agent_ids if env.agents[a].identity.role == "warehouse")
    env.states[coalition_agent].impairment = 0.8
    env.states[coalition_agent].capacity *= 0.2
    env.agents[coalition_agent].deliver_observation(env.private_observation(coalition_agent), env.ledger)
    coalition_result = ask_for_code(
        env, coalition_agent, 5,
        "temporary recovery coalition proposal", "coalition_proposed",
    )
    if coalition_result.code == "coalition_proposed":
        coalition_id = str(coalition_result.data["coalition_id"])
        # Let the explicit invitation traverse the ordinary communication
        # channel, then give the invited organization its own stressed private
        # observation.  The second LLM context independently decides whether
        # to join; the harness does not add it to coalition membership.
        env.advance()
        env.transition()
        delivered_invitees = [
            agent_id for agent_id in env.coalitions[coalition_id].invited
            if any(
                message.kind == "coalition_proposal"
                and message.payload.get("coalition_id") == coalition_id
                for message in env.agents[agent_id].inbox
            )
        ]
        if delivered_invitees:
            invitee = delivered_invitees[0]
            env.states[invitee].impairment = max(
                0.7, env.states[invitee].impairment
            )
            env.states[invitee].capacity = min(
                env.states[invitee].capacity,
                0.3 * env.states[invitee].base_capacity,
            )
            env.deliver_observations()
            ask_for_code(
                env, invitee, 5,
                "independent response to coalition invitation", "coalition_joined",
            )

    # A clean second environment isolates the route-failure/replanning probe
    # from the earlier quote conversation. The first target is unreachable;
    # after observing that tool result, the same persistent agent receives a
    # new explicit need message and replans to a reachable target.
    route_env = LogisticsEnvironment(ScenarioConfig(
        application="commercial", seed=905, horizon=8, n_agents=10,
        private_information=1.0, objective_misalignment=1.0,
        communication="reliable", disruption="nominal", decision_interval=2,
    ))
    route_env.transition()
    route_env.deliver_observations()
    source = next(a for a in route_env.agent_ids if route_env.agents[a].identity.role == "supplier")
    route_targets = [a for a in route_env.agent_ids if route_env.agents[a].identity.role == "retailer"]
    first_target, second_target = route_targets[:2]

    def deliver_need(sender: str, message_id: str) -> None:
        message = Message(
            message_id=message_id, sender=sender, recipient=source, kind="need",
            payload={"quantity": 3.0, "urgency": "critical"},
            sent_step=route_env.step_index, deliver_step=route_env.step_index,
        )
        route_env.ledger.append(route_env.step_index, "message", sender, asdict(message), private_to=source)
        route_env.agents[source].deliver_message(message)

    public_by_id = {row["agent_id"]: row for row in route_env.public_identities()}
    deliver_need(first_target, "M_STAGE1_ROUTE_CLOSED")
    route_env.physical_edges.discard((source, first_target))
    # Route execution belongs to the local-plan option. Option 6 is explicitly
    # coalition reallocation and its private affordance intentionally narrows
    # to coalition actions when no proposal has been delivered.
    ask_for_code(
        route_env, source, int(CoordinationOption.CONTINUE),
        "dispatch before route validation", "no_route",
        [public_by_id[first_target]],
    )
    deliver_need(second_target, "M_STAGE1_ROUTE_RECOVERY")
    ask_for_code(
        route_env, source, int(CoordinationOption.CONTINUE),
        "replan after failed route", "shipment_scheduled",
        [public_by_id[second_target]],
    )

    application_results = []
    for application, n_agents, seed in (("commercial", 5, 903), ("humanitarian", 6, 904)):
        config = ScenarioConfig(
            application=application, seed=seed, horizon=9, n_agents=n_agents,
            private_information=0.8, objective_misalignment=0.8,
            communication="reliable", disruption="moderate", decision_interval=3,
        )
        runner = EpisodeRunner(config, "autonomous_fixed_comm", planner=planner)
        result = runner.run("stage1-%s" % application)
        application_results.append({
            "application": application, "run_id": result.run_id,
            "metrics": result.metrics, "agent_metrics": result.agent_metrics,
            "planner_metrics": result.planner_metrics,
            "time_series": result.time_series,
        })
        run_dir = output_dir / application
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "episode.json").write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        runner.env.ledger.write_jsonl(run_dir / "events.jsonl.gz")

    codes = [row.get("tool_result", {}).get("code") for row in records]
    record = {
        "status": "complete",
        "model_identifier": model_id,
        "model_revision": revision,
        "prompt_template_revision": PLANNER_PROMPT_REVISION,
        "negotiation_and_revision": records,
        "applications": application_results,
        "checks": {
            "real_llm": True,
            "private_cost_contract_attempt": any(
                row["label"].startswith("private-cost seller offer")
                and row.get("tool_result", {}).get("code") == "offer_submitted"
                for row in records
            ),
            "rejection_observed": "offer_rejected" in codes,
            "counteroffer_observed": "offer_countered" in codes,
            "coalition_proposal_observed": "coalition_proposed" in codes,
            "coalition_observed": "coalition_joined" in codes,
            "independent_coalition_join_observed": "coalition_joined" in codes,
            "failed_action_observed": "no_route" in codes,
            "replan_observed": any(row["label"].startswith("replan after failed route") and row["tool_result"]["code"] == "shipment_scheduled" for row in records),
            "commercial_complete": application_results[0]["metrics"]["conservation_error"] < 1e-8,
            "humanitarian_complete": application_results[1]["metrics"]["conservation_error"] < 1e-8,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage1_agentic_smoke.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    required = [
        "real_llm", "private_cost_contract_attempt", "rejection_observed",
        "counteroffer_observed", "coalition_proposal_observed",
        "independent_coalition_join_observed", "failed_action_observed",
        "replan_observed", "commercial_complete", "humanitarian_complete",
    ]
    missing = [name for name in required if not record["checks"][name]]
    if missing:
        raise RuntimeError("Stage 1 requirements not observed: %s" % ", ".join(missing))
    return record
